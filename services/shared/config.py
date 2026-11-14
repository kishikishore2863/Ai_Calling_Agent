"""Central configuration, loaded from environment variables / .env.

Every model endpoint is OpenAI-compatible, so swapping a local dev backend
(Ollama on a laptop) for a production one (vLLM on a GPU box) is a config change.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LiveKit (media server) ---
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"

    # --- Telephony ---
    sip_outbound_trunk_id: str = ""  # created by scripts/setup_trunk.py
    transfer_phone_number: str = ""  # human sales rep for warm transfer (E.164), optional
    ring_timeout_s: int = 30
    max_call_duration_s: int = 600

    # --- LLM (OpenAI-compatible: Ollama / vLLM / llama.cpp server) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:3b"
    llm_api_key: str = "not-needed"
    llm_temperature: float = 0.4
    # post-call lead extraction model (not latency sensitive, can be bigger); empty = LLM_MODEL
    extraction_model: str = ""

    # --- Speech server (services/speech/server.py, OpenAI-compatible) ---
    speech_base_url: str = "http://localhost:8001/v1"
    stt_language: str = "en"
    tts_voice: str = "af_heart"  # Kokoro voice; 'hf_alpha' / 'hm_omega' for Hindi
    tts_speed: float = 1.0

    # --- Agent ---
    agent_name: str = "outbound-caller"
    agent_profile: str = "default"  # key in services/agent/profiles.py
    data_dir: Path = Path("data/calls")

    # --- Speech server internals (only used by the speech server process) ---
    whisper_model: str = "base.en"  # dev: base.en / small.en; GPU: large-v3-turbo
    whisper_device: str = "auto"  # auto | cpu | cuda
    whisper_compute_type: str = "int8"  # int8 on CPU, float16 on GPU
    kokoro_model_path: Path = Path("models/kokoro/kokoro-v1.0.int8.onnx")
    kokoro_voices_path: Path = Path("models/kokoro/voices-v1.0.bin")


@lru_cache
def get_settings() -> Settings:
    return Settings()
