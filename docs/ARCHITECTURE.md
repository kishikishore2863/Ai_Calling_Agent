# Outbound AI Voice Calling Agent — High-Level Design & Architecture

**Status:** Draft v0.1 (pre-development) · **Date:** 2026-11-14

---

## 1. Goals

| # | Goal | Success metric |
|---|------|----------------|
| G1 | Place **outbound** phone calls automatically from a contact list (campaign) | N concurrent calls, retries, calling windows respected |
| G2 | Hold a **natural, real-time voice conversation** using **open-source / self-hosted models** | Voice-to-voice latency **< 1.0 s** (p50), < 1.5 s (p95) |
| G3 | **Extract structured leads** from each conversation | ≥ 90% field accuracy on a labelled test set; every call gets a disposition |
| G4 | Push qualified leads to a CRM / dashboard | Lead visible in < 60 s after hang-up |
| G5 | Be compliant (consent, DNC, recording disclosure, AI disclosure) | Zero calls to DNC numbers; audit log for every call |

**Non-goals (v1):** inbound call handling, video, human-agent softphone UI (only warm transfer to an existing number), training models from scratch.

---

## 2. System Context

```
                ┌──────────────────────┐
  Admin / Sales │  Web Dashboard       │  upload leads, create campaigns,
  team  ───────►│  (React / Next.js)   │  watch live calls, review leads
                └──────────┬───────────┘
                           │ REST / WebSocket
                ┌──────────▼───────────┐          ┌──────────────────┐
                │   Voice Agent        │◄────────►│ CRM (HubSpot /   │
                │   Platform (ours)    │ webhooks │ Zoho / Salesforce│
                └──────────┬───────────┘          │ / Google Sheets) │
                           │ SIP / RTP            └──────────────────┘
                ┌──────────▼───────────┐
                │  SIP Trunk provider  │  (Telnyx / Twilio Elastic SIP /
                │  (PSTN gateway)      │   Plivo / local carrier)
                └──────────┬───────────┘
                           │ PSTN
                       ☎  Prospect
```

> The only paid, non-open-source piece you cannot avoid is the **SIP trunk** (you need a carrier to reach real phone numbers). Everything else can be self-hosted.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                CONTROL PLANE                                     │
│                                                                                  │
│  ┌─────────────┐   ┌──────────────────┐   ┌───────────────────┐                  │
│  │ Dashboard   │──►│ API Service      │──►│ Campaign Scheduler│                  │
│  │ (Next.js)   │   │ (FastAPI)        │   │ - calling windows │                  │
│  └─────────────┘   │ - campaigns      │   │ - DNC scrub       │                  │
│                    │ - contacts       │   │ - retry policy    │                  │
│                    │ - leads, calls   │   │ - concurrency cap │                  │
│                    └────────┬─────────┘   └─────────┬─────────┘                  │
│                             │                       │ enqueue call jobs          │
│                    ┌────────▼─────────┐   ┌─────────▼─────────┐                  │
│                    │ PostgreSQL       │   │ Redis             │                  │
│                    │ (system of record│   │ (job queue, live  │                  │
│                    └──────────────────┘   │  call state, pub/sub)                │
│                                           └─────────┬─────────┘                  │
│                                                     │                            │
│                                           ┌─────────▼─────────┐                  │
│                                           │ Dialer /          │                  │
│                                           │ Call Orchestrator │                  │
│                                           └─────────┬─────────┘                  │
└─────────────────────────────────────────────────────┼────────────────────────────┘
                                                      │ "dial +91XXXX, agent=X"
┌─────────────────────────────────────────────────────┼────────────────────────────┐
│                                MEDIA PLANE          ▼                            │
│   SIP trunk ◄──SIP/RTP──► ┌───────────────────────────────┐                      │
│                           │ Media Server                  │                      │
│                           │ LiveKit Server + LiveKit SIP  │──► Egress: recording │
│                           │ (alt: FreeSWITCH / Asterisk)  │     → MinIO (S3)     │
│                           └──────────────┬────────────────┘                      │
│                                          │ audio frames (WebRTC/RTP)             │
│                           ┌──────────────▼────────────────────────────────────┐  │
│                           │ Voice Agent Worker (Python, LiveKit Agents/Pipecat)│  │
│                           │                                                    │  │
│                           │  audio in ─► VAD ─► STT ─► Turn detect ─► LLM ─┐   │  │
│                           │                                                │   │  │
│                           │  audio out ◄─────────── TTS ◄──────────────────┘   │  │
│                           │                                                    │  │
│                           │  Conversation State Machine + Tools                │  │
│                           │  (update_lead, end_call, transfer, callback, DNC)  │  │
│                           └──────────────┬─────────────────────────────────────┘  │
└──────────────────────────────────────────┼───────────────────────────────────────┘
                                           │ gRPC/HTTP
┌──────────────────────────────────────────┼───────────────────────────────────────┐
│                     MODEL SERVING (GPU)  ▼                                       │
│  ┌──────────────────┐  ┌──────────────────────┐  ┌───────────────────────────┐   │
│  │ STT server       │  │ LLM server (vLLM)    │  │ TTS server                │   │
│  │ faster-whisper / │  │ Qwen / Llama /       │  │ Kokoro / Piper /          │   │
│  │ Parakeet         │  │ Mistral (8–14B)      │  │ Indic-Parler (Indian lang)│   │
│  └──────────────────┘  └──────────────────────┘  └───────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                           │ call.ended event
┌──────────────────────────────────────────▼───────────────────────────────────────┐
│                     POST-CALL PIPELINE (async workers)                           │
│  transcript ─► Lead Extractor (LLM + JSON schema) ─► Validator ─► Lead Scorer     │
│             ─► Dedupe / merge ─► Postgres ─► CRM Sync ─► Review queue (low conf) │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design

### 4.1 Campaign & Contact Management
- Upload contacts (CSV / API) → normalize phone numbers to **E.164** (`phonenumbers` lib).
- Campaign = { script/agent profile, contact list, calling window per timezone, max concurrency, retry policy, caller ID(s) }.
- **DNC scrub** before every dial (internal DNC list + national registry where applicable).

### 4.2 Campaign Scheduler
- Runs every few seconds; picks eligible contacts (`status = pending`, inside calling hours, retries left, not DNC).
- Enforces **global & per-campaign concurrency** and **per-caller-ID rate limits** (avoid spam flagging).
- Enqueues `CallJob` into Redis.

### 4.3 Dialer / Call Orchestrator
- Consumes `CallJob`, creates a LiveKit room, dispatches an agent worker to it, and asks LiveKit SIP to dial the number through the outbound trunk.
- Tracks call lifecycle: `queued → dialing → ringing → answered | no_answer | busy | failed → in_progress → completed`.
- Handles **answering-machine detection (AMD)**: hang up or leave a voicemail drop.
- Retry policy, e.g. no-answer → retry after 2 h, max 3 attempts.

### 4.4 Media Server — **Recommended: LiveKit (Apache-2.0) + LiveKit SIP**
| Option | Pros | Cons |
|--------|------|------|
| **LiveKit + LiveKit SIP + LiveKit Agents** ✅ | Outbound SIP built in, Python agent framework with VAD/turn detection/interruptions, recording via Egress, scales horizontally | Newer than Asterisk; another service to run |
| FreeSWITCH + `mod_audio_stream` + Pipecat | Battle-tested telephony, rich dialplan | More glue code, you build barge-in/turn logic yourself |
| Asterisk + ARI `externalMedia` / AudioSocket + Pipecat | Very common, built-in AMD | Same glue-code cost; 8 kHz audio handling |

### 4.5 Voice Agent Worker (the real-time loop)
Cascaded pipeline (STT → LLM → TTS). Chosen over speech-to-speech models because it gives **control, tool-calling, and transcripts for lead extraction**.

| Stage | Open-source choice | Notes |
|-------|--------------------|-------|
| VAD | **Silero VAD** | Detects speech start/stop; enables barge-in |
| Turn detection | LiveKit open-weights turn-detector model (or VAD silence + heuristics) | Prevents cutting the user off mid-sentence |
| STT (streaming) | **faster-whisper** (Whisper large-v3-turbo) · **NVIDIA Parakeet** (English, very fast) · **AI4Bharat IndicConformer** (Indian languages) | Phone audio is 8 kHz — benchmark on real call audio |
| LLM | **Qwen**, **Llama**, or **Mistral Small** family, 7–14B, instruct, served by **vLLM** | Needs good tool-calling + low TTFT; check each model's licence for commercial use |
| TTS (streaming) | **Kokoro-82M** (Apache-2.0, fast) · **Piper** (CPU-friendly) · **Indic Parler-TTS** (Indian langs) | ⚠️ XTTS-v2 and F5-TTS weights are **non-commercial** licences |
| Noise / echo | Krisp-free alternative: **RNNoise** / DeepFilterNet | Optional, helps STT accuracy |

**Interruption (barge-in):** when VAD detects user speech while TTS is playing → stop TTS playback, truncate the assistant message in history to what was actually spoken, continue listening.

**Latency budget (target voice-to-voice ≈ 800 ms):**
```
end-of-speech detection   200–300 ms
STT final transcript      100–200 ms
LLM time-to-first-token   150–300 ms   (stream tokens; start TTS on first sentence)
TTS time-to-first-audio    80–150 ms
network / jitter buffer    50–100 ms
```
Tricks: stream everything, sentence-chunk LLM output into TTS, keep prompts short and use vLLM prefix caching, pre-generate the greeting audio, play a short filler ("Sure, one sec…") before slow tool calls.

### 4.6 Conversation Engine
- **Agent profile** (per campaign): persona, company info, goal, qualification questions, objection-handling notes, compliance lines.
- **State machine** wraps the LLM so the call stays on track:

```
 GREETING ─► DISCLOSURE (AI + recording) ─► PERMISSION ("is now a good time?")
     │                                           │ no → SCHEDULE_CALLBACK ─► CLOSE
     ▼ yes
 PITCH ─► QUALIFY (Q1..Qn) ◄─► OBJECTION_HANDLING
     │
     ├─ interested ─► CAPTURE_DETAILS ─► BOOK_MEETING / TRANSFER_TO_HUMAN ─► CLOSE
     ├─ not interested ─► CLOSE
     └─ "don't call me" ─► MARK_DNC ─► CLOSE
```

- **LLM tools (function calls) available during the call:**

| Tool | Purpose |
|------|---------|
| `update_lead(field, value)` | Real-time slot filling (name, email, budget, …) |
| `schedule_callback(datetime)` | Book a callback slot |
| `book_meeting(slot)` | Calendar integration (Cal.com — open source) |
| `transfer_to_human()` | SIP REFER / warm transfer to sales rep |
| `mark_dnc()` | Add to do-not-call list immediately |
| `end_call(disposition)` | Polite hang-up with outcome |

- **Guardrails:** max call duration, max turns, refuse off-topic, never invent prices/offers (only from agent profile), silence timeout (“Are you still there?” → hang up).

### 4.7 Lead Extraction (two-tier)

**Tier 1 — Real-time (during call):** LLM calls `update_lead` as the prospect shares info. Fast, but may miss things.

**Tier 2 — Post-call (source of truth):**
1. Call ends → `call.ended` event with full transcript (timestamped, speaker-labelled).
2. **Extractor LLM** (can be a bigger model, e.g. 14–32B, since it’s not latency-sensitive) with **JSON-schema-constrained output** (vLLM guided decoding / `xgrammar` / `outlines`).
3. **Validator** (Pydantic): email/phone format, date parsing, enums.
4. **Merge** Tier 1 + Tier 2 values; keep per-field **confidence** + **evidence quote** from transcript.
5. **Scorer**: rules + LLM → `hot / warm / cold / disqualified`.
6. Low-confidence or conflicting fields → **human review queue** in dashboard.
7. Push to CRM via webhook / connector.

**Lead schema (example):**
```json
{
  "call_id": "uuid",
  "contact_id": "uuid",
  "disposition": "interested | not_interested | callback | wrong_number | voicemail | dnc | no_answer",
  "lead_score": "hot | warm | cold | disqualified",
  "fields": {
    "full_name":        {"value": "Ravi Kumar", "confidence": 0.95, "evidence": "This is Ravi Kumar"},
    "email":            {"value": "ravi@acme.com", "confidence": 0.80, "evidence": "..."},
    "company":          {"value": "Acme Pvt Ltd", "confidence": 0.90, "evidence": "..."},
    "role":             {"value": "Operations Head", "confidence": 0.70, "evidence": "..."},
    "is_decision_maker":{"value": true, "confidence": 0.60, "evidence": "..."},
    "budget":           {"value": "5-10 lakh", "confidence": 0.50, "evidence": "..."},
    "timeline":         {"value": "next quarter", "confidence": 0.75, "evidence": "..."},
    "need / pain_points":{"value": ["manual follow-ups"], "confidence": 0.85, "evidence": "..."},
    "callback_at":      {"value": "2026-11-18T15:00:00+05:30", "confidence": 0.90, "evidence": "..."}
  },
  "objections": ["price too high"],
  "sentiment": "positive | neutral | negative",
  "summary": "2–3 sentence call summary",
  "next_action": "send brochure by email; call back Tue 3pm"
}
```
> Tip: spelled-out emails over the phone are error-prone (“r-a-v-i at acme dot com”). Have the agent **read back** emails/numbers for confirmation, and optionally send an SMS/WhatsApp to confirm.

### 4.8 Data Model (PostgreSQL)

```
campaigns (id, name, agent_profile_id, status, window_start, window_end, tz,
           max_concurrency, retry_policy, caller_ids[])
agent_profiles (id, name, system_prompt, voice, language, questions jsonb, tools[])
contacts  (id, campaign_id, phone_e164, name, meta jsonb, status, attempts, next_attempt_at)
dnc_list  (phone_e164, reason, source, created_at)
calls     (id, contact_id, campaign_id, sip_call_id, status, disposition,
           started_at, answered_at, ended_at, duration_s, recording_url, amd_result, cost)
call_turns(id, call_id, speaker, text, start_ms, end_ms, latency_ms)
leads     (id, call_id, contact_id, score, fields jsonb, summary, sentiment,
           review_status, crm_sync_status, crm_id)
audit_log (id, entity, entity_id, action, actor, payload jsonb, at)
```
Recordings → **MinIO** (S3-compatible). Transcripts optionally indexed in **pgvector** for search/analytics later.

### 4.9 Dashboard
- Campaigns: create, upload CSV, start/pause, live stats (dialed / connected / leads).
- Live calls: real-time transcript stream (WebSocket), option to listen in / take over.
- Leads: table with filters, score, review queue, export CSV, CRM sync status.
- Call detail: recording player + transcript + extracted fields with evidence highlights.

### 4.10 Observability
- **Metrics** (Prometheus + Grafana): concurrent calls, answer rate, per-stage latency (STT/LLM/TTS), GPU utilisation, error rates.
- **LLM tracing**: **Langfuse** (open source, self-hosted) — prompts, tool calls, extraction outputs.
- **Logs**: structured JSON → Loki.
- **Quality loop**: sample N calls/week, human-label leads, measure extraction accuracy, iterate prompts.

---

## 5. End-to-End Call Sequence

```
Scheduler     Orchestrator    LiveKit SIP     SIP Trunk     Prospect     Agent Worker        Post-call
   │ CallJob       │               │              │             │              │                  │
   │──────────────►│ create room + dispatch agent ───────────────────────────►│ (joins, preloads │
   │               │──CreateSIPParticipant──►│ INVITE ──────►│ ring          │  greeting audio) │
   │               │               │              │◄── answer ──│              │                  │
   │               │◄── answered ──│◄── 200 OK ───│             │              │                  │
   │               │               │══════ RTP audio ═══════════╪═════════════►│ AMD check        │
   │               │               │              │             │◄── greeting ─│                  │
   │               │               │              │             │── speech ───►│ VAD→STT→LLM→TTS  │
   │               │               │              │             │◄── reply ────│ update_lead(...) │
   │               │               │              │             │     ...      │                  │
   │               │               │              │             │◄── goodbye ──│ end_call(disp)   │
   │               │◄── room closed / call.ended event ─────────────────────────│                  │
   │               │──────────────────────────────────────────────────────────────────────────────►│
   │               │                                             transcript → extract → score → CRM │
```

---

## 6. Tech Stack Summary

| Layer | Choice | Licence |
|-------|--------|---------|
| Language | Python 3.12 (backend + agent), TypeScript (dashboard) | — |
| API | FastAPI + SQLAlchemy + Alembic | MIT |
| Queue / state | Redis + arq (or Celery) | BSD |
| DB | PostgreSQL 16 (+ pgvector) | PostgreSQL |
| Object storage | MinIO | AGPL-3.0 (use as service; or any S3) |
| Media / SIP | LiveKit Server + LiveKit SIP + Egress | Apache-2.0 |
| Agent framework | LiveKit Agents (alt: Pipecat) | Apache-2.0 / BSD-2 |
| VAD | Silero VAD | MIT |
| STT | faster-whisper / Parakeet / IndicConformer | MIT / CC-BY-4.0 / MIT — verify per model |
| LLM serving | vLLM | Apache-2.0 |
| LLM | Qwen / Llama / Mistral instruct (7–14B realtime, 14–32B extraction) | Varies — **verify commercial terms** |
| TTS | Kokoro / Piper / Indic Parler-TTS | Apache-2.0 / MIT / Apache-2.0 — verify per voice |
| Calendar | Cal.com (self-hosted) | AGPL-3.0 |
| Observability | Prometheus, Grafana, Loki, Langfuse | Apache/AGPL/MIT |
| Dashboard | Next.js + Tailwind + shadcn/ui | MIT |
| Deploy | Docker Compose (MVP) → Kubernetes (scale) | — |

---

## 7. Infrastructure & Sizing (rough — benchmark in Phase 0)

| Concurrent calls | Suggested setup |
|------------------|-----------------|
| 1–10 (MVP) | 1× 24 GB GPU (L4 / A10G / RTX 4090) running STT + 8B LLM (AWQ/FP8) + Kokoro; 1 CPU VM for LiveKit + API + DB |
| 10–50 | Separate GPUs: LLM on 1× 48–80 GB (L40S / A100), STT + TTS on 1× 24 GB; 2–3 agent worker pods |
| 50+ | Horizontal: multiple vLLM replicas behind a router, STT/TTS autoscaled, LiveKit cluster |

Key rule: **the LLM is usually the bottleneck** — measure TTFT under concurrent load, not single-call.

Deployment topology: keep media server + agent workers + GPU servers in the **same region/data centre** as the SIP trunk’s point of presence to minimise latency.

---

## 8. Compliance & Safety (must-have before going live)

- **Consent**: only call contacts with documented opt-in. Store consent source per contact.
- **DNC**: scrub against national registry (e.g. India **NCPR/TRAI DLT** rules and designated number series for promotional calls; US **TCPA** — FCC treats AI-generated voices as “artificial voice”, requiring prior express consent) + internal DNC; honour opt-out instantly via `mark_dnc`.
- **Calling hours**: enforce per-recipient timezone.
- **Disclosure**: agent states it is an AI assistant and that the call is recorded, at the start.
- **Data protection**: encrypt recordings at rest, retention policy (e.g. 90 days), PII access control, India **DPDP Act** / GDPR where applicable.
- **Audit log** for every call and lead change.

> Get a legal review for your target country before launch — the rules change frequently.

---

## 9. Suggested Repository Layout

```
cloude_call_agent/
├── docs/                    # this document, ADRs
├── services/
│   ├── api/                 # FastAPI: campaigns, contacts, leads, webhooks
│   ├── scheduler/           # campaign scheduler
│   ├── orchestrator/        # dialer: LiveKit SIP calls, lifecycle, retries
│   ├── agent/               # LiveKit Agents worker: pipeline, state machine, tools
│   ├── postcall/            # extraction, scoring, CRM sync workers
│   └── shared/              # models, schemas (Pydantic), db, config
├── models/                  # model-serving configs (vLLM, STT, TTS docker files)
├── dashboard/               # Next.js app
├── infra/
│   ├── docker-compose.yml
│   ├── livekit/             # livekit.yaml, sip.yaml, trunk configs
│   └── k8s/                 # later
├── eval/                    # labelled transcripts, extraction accuracy tests,
│                            # latency benchmarks, simulated-caller tests
└── tests/
```

---

## 10. Phased Roadmap

| Phase | Scope | Exit criteria |
|-------|-------|---------------|
| **0. Spike (1–2 wks)** | Local LiveKit + one agent; call your own phone via SIP trunk; STT→LLM→TTS loop; measure latency; pick models | Coherent 2-min call, p50 voice-to-voice < 1.2 s |
| **1. MVP (3–4 wks)** | Postgres schema, CSV upload, scheduler + dialer, one agent profile, recording, post-call extraction, simple leads table | 100 test calls run unattended, leads extracted with ≥ 85% accuracy |
| **2. Production-ready (3–4 wks)** | AMD/voicemail, retries, DNC, real-time `update_lead`, transfer to human, callbacks, CRM sync, review queue, observability | Pilot campaign with real (consented) contacts |
| **3. Scale & quality** | Multi-language, A/B scripts, fine-tune LLM on good transcripts, autoscaling, analytics | Cost/lead and conversion targets met |

**Testing strategy:** build a **simulated caller** (a second LLM + TTS that plays a prospect persona) to run hundreds of automated test conversations without dialling real numbers.

---

## 11. Key Risks

| Risk | Mitigation |
|------|-----------|
| High latency → robotic feel | Streaming everywhere, smaller LLM for realtime, co-located GPUs, filler phrases |
| STT errors on 8 kHz / accented / noisy audio | Benchmark on real calls; domain vocabulary prompts; read-back confirmations |
| LLM hallucinating offers/prices | Strict agent profile, tool-only facts, output guardrails |
| Calls flagged as spam | Caller-ID rotation limits, registered/branded caller ID, low per-number volume |
| Licence issues with models | Licence register in `docs/`; avoid non-commercial weights |
| Legal / regulatory | Consent records, DNC, disclosure, legal review |

---

## 12. Open Decisions (need answers before Phase 0)

1. **Target country & languages** (English only? Hindi / Tamil / Telugu / other?) — drives STT/TTS choice and compliance.
2. **SIP trunk provider** and caller-ID numbers.
3. **Hosting**: own GPU server vs. cloud GPU (AWS/GCP/RunPod/E2E Networks)?
4. **Target concurrency** for v1 (e.g. 5, 20, 100 calls at once)?
5. **Lead fields & qualification questions** for the first campaign (industry/use case).
6. **CRM** to integrate with (or just dashboard + CSV export for MVP)?
7. **Media stack**: LiveKit (recommended) vs. FreeSWITCH/Asterisk (if the team already knows it).
