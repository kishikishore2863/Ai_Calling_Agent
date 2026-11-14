"""Outbound voice agent worker (LiveKit Agents).

Pipeline per call:  Silero VAD -> faster-whisper STT -> turn detector -> LLM (+tools) -> Kokoro TTS

Run modes (LiveKit CLI built in):
  uv run python -m services.agent.main console        # talk to it with your mic, no phone/server
  uv run python -m services.agent.main dev            # worker connected to LiveKit, hot reload
  uv run python -m services.agent.main start          # production
  uv run python -m services.agent.main download-files # fetch VAD / turn-detector weights

Calls are started by dispatching this agent with job metadata, see scripts/dial.py:
  {"phone_number": "+9198XXXXXXXX", "contact_name": "Ravi Kumar", "profile": "default"}
Without a phone_number the agent just joins the room (playground / console testing).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from dotenv import load_dotenv
from google.protobuf.duration_pb2 import Duration
from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatMessage,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    get_job_context,
    room_io,
)
from livekit.agents import inference
from livekit.plugins import openai, silero

from services.agent.latency import LatencyTracker
from services.agent.profiles import PROFILES, AgentProfile, build_instructions, greeting_text
from services.postcall.extract import extract_lead
from services.shared.config import get_settings

load_dotenv()
log = logging.getLogger("outbound-agent")
settings = get_settings()

LeadField = Literal[
    "full_name", "email", "phone", "company", "role", "is_decision_maker", "team_size",
    "current_solution", "pain_points", "timeline", "budget", "interest_level", "notes",
]
Disposition = Literal[
    "interested", "demo_booked", "callback", "not_interested", "wrong_number", "voicemail",
    "do_not_call", "transferred", "other",
]


@dataclass
class CallState:
    call_id: str
    profile: str
    phone_number: str | None = None
    contact_name: str | None = None
    # live (tier-1) lead capture; tier-2 post-call extraction comes in Phase 1
    lead: dict[str, str] = field(default_factory=dict)
    disposition: str | None = None
    summary: str | None = None
    callback_at: str | None = None
    callback_reason: str | None = None
    do_not_call: bool = False
    dial_started_at: float | None = None
    answered_at: float | None = None
    ended_at: float | None = None
    sip_error: str | None = None


class LeadAgent(Agent):
    def __init__(self, profile: AgentProfile, contact_name: str | None) -> None:
        super().__init__(instructions=build_instructions(profile, contact_name))

    @function_tool()
    async def update_lead(self, context: RunContext[CallState], field: LeadField, value: str) -> str:
        """Save one piece of information the prospect just told you about themselves or their
        business. Call it as soon as you learn something useful.

        Args:
            field: Which lead field this information belongs to.
            value: The value, short and normalised (e.g. email in lowercase, '5 people').
        """
        lead = context.userdata.lead
        lead[field] = f"{lead[field]}; {value}" if field in ("pain_points", "notes") and field in lead else value
        log.info("lead[%s] = %r", field, value)
        return "saved"

    @function_tool()
    async def schedule_callback(self, context: RunContext[CallState], when: str, reason: str = "") -> str:
        """Schedule a callback when the prospect is busy or asks to be called later.

        Args:
            when: When to call back, as the prospect said it (e.g. 'tomorrow 4 pm', 'Monday morning').
            reason: Optional short reason.
        """
        context.userdata.callback_at, context.userdata.callback_reason = when, reason
        log.info("callback scheduled: %s (%s)", when, reason)
        return f"Callback noted for {when}. Confirm the time with them, then end the call."

    @function_tool()
    async def mark_do_not_call(self, context: RunContext[CallState]) -> str:
        """The prospect asked not to be called again. Call this immediately."""
        context.userdata.do_not_call = True
        context.userdata.disposition = "do_not_call"
        log.info("marked do-not-call")
        return "Marked. Apologise briefly and end the call with disposition do_not_call."

    @function_tool()
    async def transfer_to_human(self, context: RunContext[CallState]) -> str:
        """Transfer the call to a human sales representative. Only when the prospect explicitly
        asks to talk to a person right now."""
        st = context.userdata
        if not settings.transfer_phone_number or not st.phone_number:
            return "Transfer is not available. Offer to schedule a callback from a specialist instead."
        await context.session.say("Sure, let me connect you to a specialist now. One moment please.")
        await context.wait_for_playout()
        st.disposition = "transferred"
        await get_job_context().transfer_sip_participant(
            st.phone_number, f"tel:{settings.transfer_phone_number}")
        return "Transferred."

    @function_tool()
    async def end_call(self, context: RunContext[CallState], disposition: Disposition, summary: str) -> str:
        """End the phone call. Use it when the conversation is finished.

        Args:
            disposition: Outcome of the call.
            summary: One sentence summary of the call and agreed next step.
        """
        st = context.userdata
        if st.disposition != "do_not_call":
            st.disposition = disposition
        st.summary = summary
        log.info("end_call disposition=%s summary=%r", disposition, summary)
        # The goodbye generated from this tool's output reuses the same speech handle,
        # so close the session (and hang up) once that speech has finished playing.
        context.speech_handle.add_done_callback(lambda _: context.session.shutdown())
        return "Now say one short, polite goodbye sentence. Do not ask anything else."


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


# shutdown_process_timeout: time allowed after hang-up for saving + post-call lead extraction
server = AgentServer(setup_fnc=prewarm, shutdown_process_timeout=60)


def _build_session(proc: JobProcess, state: CallState) -> AgentSession[CallState]:
    return AgentSession[CallState](
        userdata=state,
        vad=proc.userdata["vad"],
        stt=openai.STT(model="whisper", base_url=settings.speech_base_url, api_key="local",
                       language=settings.stt_language, use_realtime=False),
        llm=openai.LLM(model=settings.llm_model, base_url=settings.llm_base_url,
                       api_key=settings.llm_api_key, temperature=settings.llm_temperature,
                       parallel_tool_calls=False),
        tts=openai.TTS(model="kokoro", voice=settings.tts_voice, speed=settings.tts_speed,
                       base_url=settings.speech_base_url, api_key="local",
                       response_format="pcm", sample_rate=24000),
        turn_handling={
            # Pin the LOCAL end-of-turn model. Left on auto, dev mode picks LiveKit's cloud
            # model, which would send call audio to a hosted service.
            "turn_detection": inference.TurnDetector(version="v1-mini"),
            "endpointing": {"min_delay": 0.4, "max_delay": 3.0},
            # "adaptive" interruption detection is a LiveKit Cloud service; keep it local.
            "interruption": {"mode": "vad"},
            "preemptive_generation": {"enabled": True},
        },
        max_tool_steps=3,
    )


def _sip_failure(e: api.TwirpError) -> str:
    status = (e.metadata or {}).get("sip_status_code", "")
    return {"486": "busy", "600": "busy", "480": "no_answer", "408": "no_answer",
            "487": "no_answer", "404": "invalid_number"}.get(status, "failed")


async def _save_call(state: CallState, session: AgentSession, latency: LatencyTracker) -> None:
    state.ended_at = time.time()
    transcript = [
        {"role": it.role, "text": it.text_content, "interrupted": it.interrupted,
         "at": it.created_at}
        for it in session.history.items
        if isinstance(it, ChatMessage) and it.role in ("user", "assistant") and it.text_content
    ]
    record = {"call": asdict(state), "transcript": transcript, "latency": latency.summary(),
              "lead": None}
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    path = settings.data_dir / f"{state.call_id}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))  # save first, never lose a call
    log.info("saved %s | disposition=%s | live lead=%s | latency=%s",
             path, state.disposition, state.lead, record["latency"])
    try:  # Phase 1 moves this to a separate post-call worker fed by a queue
        lead = await asyncio.wait_for(extract_lead(transcript, state.lead), timeout=45)
        record["lead"] = lead.model_dump() if lead else None
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        log.info("extracted lead: %s", record["lead"] and
                 {k: v["value"] for k, v in record["lead"].items() if isinstance(v, dict) and v["value"]})
    except Exception:
        log.exception("lead extraction failed (re-run: python -m services.postcall.extract %s)", path)


@server.rtc_session(agent_name=settings.agent_name)
async def entrypoint(ctx: JobContext) -> None:
    meta = json.loads(ctx.job.metadata or "{}")
    phone = meta.get("phone_number")
    profile_key = meta.get("profile", settings.agent_profile)
    profile = PROFILES[profile_key]
    state = CallState(call_id=ctx.room.name, profile=profile_key, phone_number=phone,
                      contact_name=meta.get("contact_name"))
    ctx.log_context_fields = {"call_id": state.call_id, "phone": phone}

    session = _build_session(ctx.proc, state)
    latency = LatencyTracker()

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if isinstance(ev.item, ChatMessage):
            latency.on_message(ev.item)

    session.on("close", lambda _ev: ctx.shutdown("session closed"))
    ctx.add_shutdown_callback(lambda: _save_call(state, session, latency))

    agent = LeadAgent(profile, state.contact_name)
    # For a phone call, listen only to the SIP participant, and hang up when the session closes.
    room_options = (room_io.RoomOptions(participant_identity=phone, delete_room_on_close=True)
                    if phone else room_io.RoomOptions())

    if phone:
        if not settings.sip_outbound_trunk_id:
            raise RuntimeError("SIP_OUTBOUND_TRUNK_ID is not set; run scripts/setup_trunk.py")
        # Start the session while the phone rings so models are warm when they answer.
        session_started = asyncio.create_task(
            session.start(agent=agent, room=ctx.room, room_options=room_options))
        state.dial_started_at = time.time()
        log.info("dialing %s", phone)
        try:
            await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=settings.sip_outbound_trunk_id,
                sip_call_to=phone,
                participant_identity=phone,
                participant_name=state.contact_name or "prospect",
                wait_until_answered=True,
                ringing_timeout=Duration(seconds=settings.ring_timeout_s),
                max_call_duration=Duration(seconds=settings.max_call_duration_s),
            ))
        except api.TwirpError as e:
            state.disposition, state.sip_error = _sip_failure(e), f"{e.code}: {e.message} {e.metadata}"
            log.warning("call not connected: %s", state.sip_error)
            session_started.cancel()
            ctx.shutdown("call not answered")
            return
        await session_started
        state.answered_at = time.time()
        log.info("answered after %.1fs", state.answered_at - state.dial_started_at)
    else:
        await session.start(agent=agent, room=ctx.room, room_options=room_options)

    # Fixed greeting via TTS only (no LLM round-trip) = fastest possible first words.
    session.say(greeting_text(profile, state.contact_name), allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
