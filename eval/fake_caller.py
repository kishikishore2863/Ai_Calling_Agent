"""Simulated phone caller: joins a LiveKit room as the 'prospect' and speaks scripted lines
(synthesised by our own TTS) in real time, waiting for the agent to answer each one.

Exercises the full audio path: VAD -> STT -> turn detection -> LLM -> TTS, with no phone line.
Measures, from the caller's side, the silence between the end of each prospect line and
the start of the agent's reply (what a real caller experiences).

  # terminal 1: livekit-server --dev
  # terminal 2: uv run python -m services.agent.main dev
  # terminal 3:
  uv run python eval/fake_caller.py
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit import api, rtc  # noqa: E402

from services.shared.config import get_settings  # noqa: E402

SR = 24000
FRAME = SR // 100  # 10 ms
LINES = [
    "Yes, I have a minute. Who is this?",
    "I'm Ravi Kumar. I run sales at Acme Traders.",
    "We have six people in sales and we use Excel sheets for follow ups.",
    "I'm busy right now. Can you call me back tomorrow at four pm?",
    "Okay, thanks. Bye.",
]


class AgentListener:
    """Tracks when the agent is audibly speaking, from its published audio track."""

    def __init__(self) -> None:
        self.speaking = False
        self.started_at: float | None = None
        self.last_voice_at = 0.0

    async def run(self, track: rtc.Track) -> None:
        async for ev in rtc.AudioStream(track, sample_rate=SR, num_channels=1):
            pcm = np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32)
            if pcm.size and np.sqrt(np.mean(pcm**2)) > 300:
                now = time.perf_counter()
                if not self.speaking:
                    self.speaking, self.started_at = True, now
                self.last_voice_at = now
            elif self.speaking and time.perf_counter() - self.last_voice_at > 0.8:
                self.speaking = False

    async def wait_reply_start(self, after: float, timeout: float) -> float | None:
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            if self.started_at and self.started_at > after:
                return self.started_at
            await asyncio.sleep(0.02)
        return None

    async def wait_silence(self, timeout: float = 30) -> None:
        end = time.perf_counter() + timeout
        await asyncio.sleep(0.3)
        while time.perf_counter() < end and (
                self.speaking or time.perf_counter() - self.last_voice_at < 1.2):
            await asyncio.sleep(0.05)


async def tts(http: aiohttp.ClientSession, base_url: str, text: str, voice: str) -> np.ndarray:
    async with http.post(f"{base_url}/audio/speech",
                         json={"input": text, "voice": voice, "response_format": "pcm"}) as r:
        r.raise_for_status()
        return np.frombuffer(await r.read(), dtype=np.int16)


async def speak(source: rtc.AudioSource, pcm: np.ndarray) -> None:
    pcm = np.concatenate([pcm, np.zeros(SR // 5, dtype=np.int16)])  # trailing 200 ms
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        chunk = pcm[i:i + FRAME]
        await source.capture_frame(rtc.AudioFrame(chunk.tobytes(), SR, 1, FRAME))


async def silence(source: rtc.AudioSource, stop: asyncio.Event) -> None:
    """Keep sending silent frames like a real phone line does."""
    z = np.zeros(FRAME, dtype=np.int16).tobytes()
    while not stop.is_set():
        await source.capture_frame(rtc.AudioFrame(z, SR, 1, FRAME))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="am_michael", help="Kokoro voice for the prospect")
    ap.add_argument("--reply-timeout", type=float, default=30)
    args = ap.parse_args()
    s = get_settings()
    room_name = f"sim-{uuid.uuid4().hex[:8]}"

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http:
        clips = [await tts(http, s.speech_base_url, line, args.voice) for line in LINES]

    async with api.LiveKitAPI(s.livekit_url, s.livekit_api_key, s.livekit_api_secret) as lk:
        await lk.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
            agent_name=s.agent_name, room=room_name,
            metadata=json.dumps({"contact_name": "Ravi Kumar"})))
    token = (api.AccessToken(s.livekit_api_key, s.livekit_api_secret)
             .with_identity("prospect").with_name("Ravi Kumar")
             .with_grants(api.VideoGrants(room_join=True, room=room_name)).to_jwt())

    room = rtc.Room()
    listener = AgentListener()
    tasks: list[asyncio.Task] = []

    @room.on("track_subscribed")
    def _on_track(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            tasks.append(asyncio.create_task(listener.run(track)))

    await room.connect(s.livekit_url, token)
    source = rtc.AudioSource(SR, 1)
    await room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track("mic", source),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    print(f"joined {room_name}; waiting for the agent's greeting...")

    t0 = time.perf_counter()
    if await listener.wait_reply_start(t0, timeout=60) is None:
        sys.exit("agent never spoke (is the agent worker running?)")
    print(f"greeting started {time.perf_counter() - t0:.1f}s after joining")

    gaps = []
    for line, pcm in zip(LINES, clips):
        await listener.wait_silence()
        print(f"\nPROSPECT: {line}")
        await speak(source, pcm)
        done_at = time.perf_counter() - 0.2  # minus the trailing silence we appended
        stop = asyncio.Event()
        filler = asyncio.create_task(silence(source, stop))
        started = await listener.wait_reply_start(done_at, timeout=args.reply_timeout)
        if started is None:
            print("  agent did not reply")
        else:
            gaps.append(started - done_at)
            print(f"  agent replied after {gaps[-1] * 1000:.0f} ms of silence")
        await listener.wait_silence()
        stop.set()
        await filler

    if gaps:
        print(f"\nvoice-to-voice (caller side): p50 {statistics.median(gaps) * 1000:.0f} ms, "
              f"max {max(gaps) * 1000:.0f} ms over {len(gaps)} turns")
    await asyncio.sleep(3)
    await room.disconnect()
    for t in tasks:
        t.cancel()
    print(f"agent's saved result: {s.data_dir / (room_name + '.json')}")


if __name__ == "__main__":
    asyncio.run(main())
