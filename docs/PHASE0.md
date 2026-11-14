# Phase 0 — Spike: runbook & findings

**Goal:** one AI agent holds a coherent ~2 min call, captures a lead, and we measure latency.
**Exit criteria:** coherent call end-to-end · lead extracted · p50 voice-to-voice < 1.2 s.

---

## What's built

| Piece | File | Status |
|-------|------|--------|
| Speech server (faster-whisper STT + Kokoro TTS, OpenAI-compatible API) | `services/speech/server.py` | ✅ working |
| Voice agent (VAD → STT → turn detect → LLM + tools → TTS) | `services/agent/main.py` | ✅ working |
| Campaign script / persona / qualification questions | `services/agent/profiles.py` | ✅ placeholder product |
| Live lead tools: `update_lead`, `schedule_callback`, `mark_do_not_call`, `transfer_to_human`, `end_call` | `services/agent/main.py` | ✅ (quality depends on model size) |
| Post-call lead extraction (JSON schema + evidence check) | `services/postcall/extract.py` | ✅ working |
| Per-turn latency tracking (p50/p95 saved per call) | `services/agent/latency.py` | ✅ |
| Call record (transcript + live lead + extracted lead + latency) | `data/calls/<room>.json` | ✅ |
| Outbound dial / SIP trunk setup | `scripts/dial.py`, `scripts/setup_trunk.py` | ⏳ written, **untested** (needs SIP trunk) |
| Text-only conversation test (prompt + tools, no audio) | `eval/text_sim.py` | ✅ |
| Simulated caller (full audio path, no phone) | `eval/fake_caller.py` | ✅ |
| GPU server deployment | `infra/docker-compose.yml`, `infra/Dockerfile.*` | ⏳ written, **untested** |

All models are self-hosted and open source: Silero VAD, LiveKit turn-detector `v1-mini`
(local), faster-whisper, Qwen 2.5 (Ollama / vLLM), Kokoro-82M.

> ⚠️ **Cloud-default gotcha:** LiveKit Agents 1.8 defaults some components to LiveKit Cloud
> in dev mode (turn detector `v1`, "adaptive" interruption detection). Both are pinned to
> local in `main.py` (`TurnDetector(version="v1-mini")`, `interruption.mode="vad"`).
> Keep it that way, or call audio goes to a hosted service.

---

## Run locally (Mac, no phone line)

```bash
uv sync --extra speech
```
```bash
cp .env.example .env
```
```bash
./scripts/download_models.sh
```
```bash
ollama serve
```
```bash
ollama pull qwen2.5:3b
```

Then four terminals:

```bash
uv run --extra speech uvicorn services.speech.server:app --port 8001
```
```bash
livekit-server --dev
```
```bash
uv run python -m services.agent.main dev
```
```bash
uv run python eval/fake_caller.py
```

Talk to it yourself with your mic instead of the fake caller (needs only the speech server + Ollama):

```bash
uv run python -m services.agent.main console
```

Fast prompt/tool iteration without audio:

```bash
uv run python eval/text_sim.py
```

Re-run lead extraction on saved calls (e.g. after changing the extractor or model):

```bash
uv run python -m services.postcall.extract data/calls/*.json
```

---

## Real phone calls (next step)

Needs a **SIP trunk** and a server with a **public IP** (RTP audio can't reach a laptop behind NAT).

1. Get a SIP trunk + caller-ID number (Telnyx / Twilio Elastic SIP / Plivo / local carrier).
   For India, check TRAI rules for promotional calls before dialling real prospects.
2. On a Linux GPU server: fill `.env` (strong `LIVEKIT_API_KEY/SECRET`, `SIP_TRUNK_*`,
   `LLM_BASE_URL=http://127.0.0.1:8000/v1`, `LLM_MODEL=Qwen/Qwen2.5-7B-Instruct`), set the
   same key/secret in `infra/livekit/sip.yaml`, then:
   ```bash
   docker compose -f infra/docker-compose.yml --env-file .env up -d --build
   ```
3. Register the trunk and put the printed ID in `.env` as `SIP_OUTBOUND_TRUNK_ID`:
   ```bash
   uv run python scripts/setup_trunk.py
   ```
4. Call **your own phone** first:
   ```bash
   uv run python scripts/dial.py +91XXXXXXXXXX --name "Your Name"
   ```

---

## Findings so far (2026-11-14, Apple M1 8 GB, everything on CPU)

### Latency (simulated caller, 5 turns)

| Stage | p50 on M1 CPU | Expected on 24 GB GPU* |
|-------|---------------|------------------------|
| End-of-turn decision | ~1.2 s | ~0.5 s (0.4 s min delay + model) |
| STT (whisper base.en) | ~1.1 s | 0.1–0.2 s (large-v3-turbo) |
| LLM first token (qwen2.5 3B) | ~0.3 s (5–7 s on the cold first turn) | 0.15–0.3 s (7B, vLLM) |
| TTS first audio (Kokoro) | ~0.6–1.1 s | < 0.1 s |
| **Voice-to-voice** | **~3.3 s** (2.3–8.5 s) | **~0.9–1.2 s** |

\*GPU figures are estimates. **Measuring them is the main remaining Phase 0 task.**
Run `eval/fake_caller.py` against the GPU deployment.

### Quality

- **The live 3B agent is not good enough.** It follows the script and asks one question at a
  time, but it rarely calls `update_lead`, didn't call `schedule_callback`/`end_call` reliably,
  and sometimes invents things (a phone number; "4 pm today" instead of tomorrow).
  → Use a **7B–14B** model for the live agent (Qwen 2.5 7B/14B Instruct on vLLM).
- **Post-call extraction is what makes leads reliable.** From the same calls it recovered
  name, company, team size, current tool, pain point and callback time with quotes. The
  3B extractor still guessed some values (role, an email). The **evidence check** flags
  quotes not spoken by the prospect (`verified: false`), but it can't catch every invented
  value. → Use a bigger extraction model (`EXTRACTION_MODEL`, e.g. 14B–32B), since it's not
  latency-sensitive.
- Speech recognition of spelled-out emails is error-prone ("acme" → "accme"). The agent
  spells emails back to confirm. Later, also confirm by SMS/WhatsApp.

### Not done yet in Phase 0

- [ ] GPU server deployment + latency benchmark (target p50 < 1.2 s)
- [ ] SIP trunk + first real call to own phone
- [ ] Compare 7B vs 14B live model (tool-calling reliability vs latency)
- [ ] Answering-machine detection (LiveKit has an AMD module; it needs an LLM + STT, so check it works locally)
- [ ] Real product, script and qualification questions in `profiles.py`
