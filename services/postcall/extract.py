"""Tier-2 lead extraction: full transcript -> structured, validated lead.

Runs after the call (not latency sensitive), so it can use a bigger model than the
realtime agent. Output is constrained by a JSON schema (supported by vLLM and Ollama),
then validated with Pydantic. Tier-1 values captured live by `update_lead` are passed in
as hints and merged.

  uv run python -m services.postcall.extract data/calls/*.json   # (re-)extract saved calls in place
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from services.shared.config import get_settings

log = logging.getLogger("postcall")


class Fact(BaseModel):
    value: str | None = Field(None, description="Value stated by the prospect, null if not said")
    evidence: str | None = Field(None, description="Short exact quote from the prospect")
    # set by us, not the model: evidence really appears in the prospect's own words
    verified: bool = False


class Lead(BaseModel):
    disposition: Literal["interested", "demo_booked", "callback", "not_interested",
                         "wrong_number", "voicemail", "do_not_call", "no_conversation"]
    lead_score: Literal["hot", "warm", "cold", "disqualified"]
    full_name: Fact
    email: Fact
    company: Fact
    role: Fact
    is_decision_maker: Fact
    team_size: Fact
    current_solution: Fact
    pain_points: Fact
    timeline: Fact
    budget: Fact
    callback_time: Fact
    objections: list[str]
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str = Field(description="At most 3 short sentences summarising the call")
    next_action: str = Field(description="Concrete next step for the sales team")


SYSTEM = """You extract sales lead data from a phone call transcript between an AI sales agent
(assistant) and a prospect (user). Rules:
- Use ONLY what the prospect said. If a field was not stated, set value and evidence to null.
- Never copy information that only the agent said.
- Normalise emails spoken aloud ('ravi at acme dot com' -> 'ravi@acme.com').
- evidence must be a short exact quote from the prospect's lines.
- lead_score: hot = clear need + decision maker + timeline within 3 months;
  warm = interested but missing some of those; cold = little interest; disqualified = wrong
  person, not a fit, or asked not to be called."""


def _format_transcript(transcript: list[dict]) -> str:
    who = {"assistant": "AGENT", "user": "PROSPECT"}
    return "\n".join(f"{who.get(t['role'], t['role'])}: {t['text']}" for t in transcript)


async def extract_lead(transcript: list[dict], live_hints: dict | None = None) -> Lead | None:
    if not any(t["role"] == "user" for t in transcript):
        return None
    s = get_settings()
    client = AsyncOpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)
    call_time = datetime.fromtimestamp(transcript[0].get("at") or time.time()).astimezone()
    user = (f"The call took place on {call_time:%A %d %B %Y, %H:%M %Z}. Resolve relative times "
            f"like 'tomorrow 4 pm' to an ISO date-time using that date.\n\n"
            f"TRANSCRIPT:\n{_format_transcript(transcript)}")
    if live_hints:
        user += f"\n\nValues the agent saved during the call (verify against transcript):\n{json.dumps(live_hints)}"
    resp = await client.chat.completions.create(
        model=s.extraction_model or s.llm_model,
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "lead", "schema": Lead.model_json_schema()}},
    )
    lead = Lead.model_validate_json(resp.choices[0].message.content)
    _verify_evidence(lead, transcript)
    # live-captured value wins when extraction missed the field
    for k, v in (live_hints or {}).items():
        f = getattr(lead, k, None)
        if isinstance(f, Fact) and not f.value:
            setattr(lead, k, Fact(value=v, evidence="(captured live by agent)"))
    return lead


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _verify_evidence(lead: Lead, transcript: list[dict]) -> None:
    """Hallucination guard: a fact is 'verified' only if its quote is in the prospect's lines.
    Unverified facts with values go to the human review queue (Phase 2)."""
    said = _norm(" ".join(t["text"] for t in transcript if t["role"] == "user"))
    for name in Lead.model_fields:
        f = getattr(lead, name)
        if not isinstance(f, Fact):
            continue
        ev = _norm(f.evidence or "")
        f.verified = bool(ev) and ev in said
        if f.evidence and not f.verified:
            f.evidence = None


async def _main(paths: list[str]) -> None:
    """Extract (or re-extract) leads for saved calls and write them back into the files."""
    from pathlib import Path
    for path in paths:
        p = Path(path)
        record = json.loads(p.read_text())
        lead = await extract_lead(record["transcript"], record["call"].get("lead"))
        record["lead"] = lead.model_dump() if lead else None
        p.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(f"== {p.name}")
        print(lead.model_dump_json(indent=2) if lead else "no conversation")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main(sys.argv[1:]))
