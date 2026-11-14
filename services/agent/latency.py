"""Per-turn latency tracking — the core measurement of Phase 0.

Uses ChatMessage.metrics that LiveKit Agents fills on every conversation turn:
  user turn:      end_of_turn_delay, transcription_delay, stopped_speaking_at
  assistant turn: llm_node_ttft, tts_node_ttfb, started_speaking_at

voice-to-voice = assistant.started_speaking_at - user.stopped_speaking_at
(the silence the caller actually hears).
"""

from __future__ import annotations

import logging
import statistics

from livekit.agents import ChatMessage

log = logging.getLogger("latency")

KEYS = ("end_of_turn", "stt", "llm_ttft", "tts_ttfb", "voice_to_voice")


class LatencyTracker:
    def __init__(self) -> None:
        self.turns: list[dict[str, float]] = []
        self._last_user: dict | None = None

    def on_message(self, msg: ChatMessage) -> None:
        m = dict(msg.metrics or {})
        if msg.role == "user":
            self._last_user = m
            return
        if msg.role != "assistant":
            return

        turn: dict[str, float] = {}
        u = self._last_user or {}
        if "end_of_turn_delay" in u:
            turn["end_of_turn"] = u["end_of_turn_delay"]
        if "transcription_delay" in u:
            turn["stt"] = u["transcription_delay"]
        if "llm_node_ttft" in m:
            turn["llm_ttft"] = m["llm_node_ttft"]
        if "tts_node_ttfb" in m:
            turn["tts_ttfb"] = m["tts_node_ttfb"]
        if "started_speaking_at" in m and "stopped_speaking_at" in u:
            turn["voice_to_voice"] = m["started_speaking_at"] - u["stopped_speaking_at"]
        self._last_user = None  # next assistant msg (e.g. greeting/tool reply) isn't a reply to it
        if not turn:
            return
        self.turns.append(turn)
        log.info("turn %d | %s", len(self.turns),
                 " | ".join(f"{k} {turn[k] * 1000:.0f}ms" for k in KEYS if k in turn))

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for k in KEYS:
            vals = sorted(t[k] for t in self.turns if k in t)
            if not vals:
                continue
            p95 = vals[min(len(vals) - 1, round(0.95 * (len(vals) - 1)))]
            out[k] = {"p50_ms": round(statistics.median(vals) * 1000),
                      "p95_ms": round(p95 * 1000), "n": len(vals)}
        return out
