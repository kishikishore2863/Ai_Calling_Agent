"""Text-only conversation test: runs the real agent (prompt + tools + LLM) without audio.

Use it to iterate on the prompt / tools quickly and to check the LLM calls tools correctly.

  uv run python eval/text_sim.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit.agents import AgentSession  # noqa: E402
from livekit.plugins import openai  # noqa: E402

from services.agent.main import CallState, LeadAgent  # noqa: E402
from services.agent.profiles import PROFILES  # noqa: E402
from services.postcall.extract import extract_lead  # noqa: E402
from services.shared.config import get_settings  # noqa: E402

PROSPECT_TURNS = [
    "Yeah, I have a minute. Who is this again?",
    "Okay. I'm Ravi Kumar, I run sales at Acme Traders.",
    "We use Excel sheets and WhatsApp mostly. Follow-ups get missed all the time.",
    "We have six people in sales. I make the decision on tools.",
    "Maybe in the next month or two. Budget is around ten thousand rupees a month.",
    "Sure, send me details. My email is ravi at acme traders dot com.",
    "Yes, that's correct.",
    "Actually I'm busy now, can you call me back tomorrow at 4 pm?",
    "Okay thanks, bye.",
]


async def main() -> None:
    s = get_settings()
    state = CallState(call_id="text-sim", profile="default", contact_name=None)
    llm = openai.LLM(model=s.llm_model, base_url=s.llm_base_url, api_key=s.llm_api_key,
                     temperature=s.llm_temperature, parallel_tool_calls=False)
    async with AgentSession[CallState](llm=llm, userdata=state, max_tool_steps=3) as session:
        await session.start(LeadAgent(PROFILES["default"], None))
        for text in PROSPECT_TURNS:
            print(f"\nPROSPECT: {text}")
            t = time.perf_counter()
            result = await session.run(user_input=text)
            ms = (time.perf_counter() - t) * 1000
            for ev in result.events:
                item = ev.item
                if item.type == "function_call":
                    print(f"  [tool] {item.name}({item.arguments})")
                elif item.type == "function_call_output":
                    print(f"  [tool result{' ERROR' if item.is_error else ''}] {item.output[:200]}")
                elif item.type == "message" and item.role == "assistant":
                    print(f"AGENT ({ms:.0f}ms): {item.text_content}")
            if state.summary:  # end_call was invoked
                break

        transcript = [{"role": m.role, "text": m.text_content} for m in session.history.items
                      if m.type == "message" and m.role in ("user", "assistant") and m.text_content]

    print("\n=== LIVE (tier 1) ===")
    print("disposition:", state.disposition, "| callback:", state.callback_at)
    print("lead:", json.dumps(state.lead))

    t = time.perf_counter()
    lead = await extract_lead(transcript, state.lead)
    print(f"\n=== POST-CALL EXTRACTION (tier 2, {time.perf_counter() - t:.1f}s) ===")
    print(lead.model_dump_json(indent=2) if lead else "no conversation")


if __name__ == "__main__":
    asyncio.run(main())
