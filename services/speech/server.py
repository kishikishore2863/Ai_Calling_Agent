"""Self-hosted speech server with an OpenAI-compatible API.

  POST /v1/audio/transcriptions  -> faster-whisper (STT)
  POST /v1/audio/speech          -> Kokoro-82M via ONNX (TTS)

The LiveKit agent talks to this with the stock `livekit.plugins.openai` STT/TTS
clients pointed at `SPEECH_BASE_URL`, so no custom plugin code is needed.

Run:  uv run --extra speech uvicorn services.speech.server:app --port 8001
"""

from __future__ import annotations

import io
import logging
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from kokoro_onnx import Kokoro
from pydantic import BaseModel

from services.shared.config import get_settings

log = logging.getLogger("speech-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings = get_settings()

# Kokoro voices are prefixed by language: a=US English, b=UK English, h=Hindi, ...
KOKORO_LANG = {"a": "en-us", "b": "en-gb", "h": "hi", "e": "es", "f": "fr-fr",
               "i": "it", "j": "ja", "p": "pt-br", "z": "cmn"}
TTS_SAMPLE_RATE = 24000


class _Models:
    whisper: WhisperModel
    kokoro: Kokoro
    # CPU inference is not safely re-entrant; serialise per model (GPU deploys scale by replicas).
    stt_lock = threading.Lock()
    tts_lock = threading.Lock()


models = _Models()


@asynccontextmanager
async def lifespan(_: FastAPI):
    t = time.perf_counter()
    models.whisper = WhisperModel(settings.whisper_model, device=settings.whisper_device,
                                  compute_type=settings.whisper_compute_type)
    if not settings.kokoro_model_path.exists():
        raise RuntimeError(f"Kokoro model missing at {settings.kokoro_model_path}; "
                           "run scripts/download_models.sh")
    models.kokoro = Kokoro(str(settings.kokoro_model_path), str(settings.kokoro_voices_path))
    # Warm-up so the first real request isn't slow.
    models.whisper.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
    models.kokoro.create("Hello.", voice=settings.tts_voice, lang="en-us")
    log.info("models loaded in %.1fs (whisper=%s)", time.perf_counter() - t, settings.whisper_model)
    yield


app = FastAPI(title="speech-server", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "whisper_model": settings.whisper_model}


@app.post("/v1/audio/transcriptions", response_model=None)
def transcribe(
    file: UploadFile = File(...),
    model: str | None = Form(None),  # accepted for API compatibility; the loaded model is used
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
) -> Response | dict:
    t = time.perf_counter()
    audio = decode_audio(io.BytesIO(file.file.read()), sampling_rate=16000)
    with models.stt_lock:
        segments, _info = models.whisper.transcribe(
            audio,
            language=language or settings.stt_language,
            beam_size=1,                       # greedy: lowest latency
            condition_on_previous_text=False,  # each utterance is independent
            initial_prompt=prompt,             # domain vocabulary hint (product names etc.)
            vad_filter=False,                  # LiveKit already ran Silero VAD
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    ms = (time.perf_counter() - t) * 1000
    log.info("stt %.0fms audio=%.2fs text=%r", ms, len(audio) / 16000, text)
    if response_format == "text":
        return Response(text, media_type="text/plain")
    return {"text": text}


class SpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: str | None = None
    response_format: str = "pcm"
    speed: float = 1.0


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if not req.input.strip():
        return Response(b"", media_type="audio/pcm")
    voice = req.voice or settings.tts_voice
    lang = KOKORO_LANG.get(voice[:1], "en-us")
    t = time.perf_counter()
    try:
        with models.tts_lock:
            samples, sr = models.kokoro.create(req.input, voice=voice, speed=req.speed, lang=lang)
    except KeyError as e:
        raise HTTPException(400, f"unknown voice {voice!r}") from e
    ms = (time.perf_counter() - t) * 1000
    log.info("tts %.0fms chars=%d audio=%.2fs", ms, len(req.input), len(samples) / sr)
    headers = {"X-Processing-Ms": f"{ms:.0f}"}

    if req.response_format == "pcm":  # raw 16-bit mono, 24 kHz (what the agent requests)
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        return Response(pcm, media_type="audio/pcm", headers=headers)
    if req.response_format == "wav":
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav", headers=headers)
    raise HTTPException(400, "response_format must be 'pcm' or 'wav'")
