# Outbound AI Voice Calling Agent

Self-hosted outbound calling agent built on open-source models. It calls prospects, runs
a qualification conversation, and extracts structured leads from each call.

- Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Current phase, how to run, findings: [docs/PHASE0.md](docs/PHASE0.md)

```
services/
  agent/      LiveKit voice agent: pipeline, script/profile, live lead tools, latency tracking
  speech/     faster-whisper STT + Kokoro TTS behind an OpenAI-compatible API
  postcall/   transcript -> structured lead (JSON schema + evidence verification)
  shared/     config (.env)
scripts/      dial a number, register SIP trunk, download models
eval/         text-only conversation test, simulated audio caller
infra/        docker-compose for a GPU server (LiveKit, SIP, vLLM, speech, agent)
```
# Ai_Calling_Agent
