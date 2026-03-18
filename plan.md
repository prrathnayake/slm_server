# Implementation Plan: Local-First LLM/SLM Platform

## 1. Project Goal

Build a local-first LLM/SLM platform that acts like your own Ollama-style provider:

- One local API for all apps and agents
- Runs local models and remote models behind one interface
- Supports chat/completions, streaming, tool-calling style outputs, embeddings later
- Supports local fine-tuning and remote fine-tuning
- Lets you import externally fine-tuned models, including from Google Colab
- Registers every model automatically so it becomes callable through your API
- Exposes an OpenAI-compatible interface so your agent apps can use one client format instead of custom integrations

> Using an OpenAI-compatible API is the right design choice because modern open-source serving stacks already support that pattern. vLLM exposes an OpenAI-compatible server, TGI supports an OpenAI-compatible Messages API, and llama.cpp also has OpenAI-compatible serving paths.

---

## 2. What You Should Build

Build it as one platform with **5 core subsystems**:

1. **Gateway API** — request handling
2. **Model Runtime Layer** — inference execution
3. **Training Orchestration Layer** — training execution
4. **Model Registry** — model metadata
5. **Storage + Artifact Management** — files and checkpoints

That is the cleanest architecture because it separates concerns cleanly.

---

## 3. Recommended High-Level Architecture

### A. Gateway API

This is the only service your apps talk to.

**Responsibilities:**
- Expose `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- Auth
- Request validation
- Streaming
- Model routing
- Usage metrics
- Logging
- Failover between local and remote backends

**Recommended stack:**
- FastAPI
- Pydantic
- Uvicorn/Gunicorn
- SSE streaming first, WebSocket optional

**Why:**
- FastAPI is a good fit for typed APIs and streaming-style inference gateways
- You want one stable API contract even if backends change later

### B. Model Runtime Layer

This is where models actually run. You should support multiple runtime backends instead of locking to one.

**Option 1 — vLLM backend**
- Best for: high-throughput transformer serving, OpenAI-compatible serving, serving base models and LoRA adapters, future scale-up to multiple GPUs or servers
- vLLM supports an OpenAI-compatible server and LoRA serving, which matches your goal well.

**Option 2 — llama.cpp backend**
- Best for: GGUF models, laptop/desktop local inference, CPU or smaller GPU memory use, quantized local models
- llama.cpp is designed for efficient local inference on a wide range of hardware.

**Option 3 — TGI backend**
- Best for: production-style Hugging Face serving, OpenAI-compatible messages API, structured model deployment in containerized environments
- TGI is a dedicated deployment stack for open-source LLMs and supports OpenAI-compatible Messages API.

### C. Training Orchestration Layer

**Handles:**
- Dataset upload
- Dataset validation
- Fine-tune job creation
- Local vs remote execution
- Job status
- Artifact retrieval
- Model registration after training

**Recommended stack:**
- Python worker service
- Celery / RQ / Dramatiq
- Redis for queue
- Training launched through containers or isolated job runners

### D. Model Registry

This is critical.

**Store:**
- model id
- display name
- backend type (local or remote)
- base model
- fine-tune type
- prompt template / chat template
- tokenizer path
- quantization format
- context length
- tool-calling support
- current status
- artifact paths
- version
- tags

**Use:**
- Postgres if you want a serious system
- SQLite if starting small

### E. Artifact Storage

**Store:**
- Raw datasets
- Processed datasets
- Training configs
- Logs
- Adapters
- Merged models
- Tokenizer files
- Exported GGUF
- Uploaded zip models from Colab

**Use:**
- Local filesystem first
- Later S3-compatible object storage if needed

---

## 4. Core Feature Set

### Feature 1: OpenAI-Compatible Unified API

**What it does:** Makes your local provider usable like OpenAI/Anthropic style from your agent app.

**Endpoints to support first:**
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings` (later)
- `POST /v1/responses` (later, if newer-style compatibility needed)
- `GET /health`
- `GET /metrics`

**Scenario A — Pass-through gateway**
- Your API receives request and forwards to backend runtime with minimal translation.
- Best when: backend already speaks OpenAI format (vLLM or TGI)

**Scenario B — Normalization gateway**
- Your API uses your own internal request schema, then converts to each backend's format.
- Best when: you will support mixed backends, you want backend independence

> **Recommended: Scenario B.**

### Feature 2: Multi-Backend Runtime Support

**What it does:** Lets one platform run GGUF models, Transformers models, local models, remote models, future hosted backends.

**Backend A — Local llama.cpp runtime**
- Use for: quantized GGUF SLMs, laptop inference, always-on small models

**Backend B — Local vLLM runtime**
- Use for: full HF transformer models, larger GPUs, better throughput, LoRA-serving

**Backend C — Remote HTTP backend**
- Use for: remote GPU servers, Colab-style external inference servers, cloud VMs

**Backend D — Remote SSH-managed backend**
- Use for: your own rented GPU boxes, secure self-managed machines

**Recommended order:**
1. local llama.cpp
2. local vLLM
3. remote HTTP
4. remote SSH orchestration

### Feature 3: Model Registry and Dynamic Model Onboarding

**What it does:** Every model becomes discoverable and callable automatically.

**Required metadata:**
- `model_id`
- `provider` = local-provider
- `runtime_backend` = llama_cpp | vllm | tgi | remote_http
- `model_format` = gguf | safetensors | adapter
- `base_model`
- `specialization` = reasoning | planning | tool-calling | personality | coder
- `status` = ready | loading | training | failed
- `source` = local_train | remote_train | imported_zip
- `capabilities` = stream, tools, json_mode, embeddings
- `version`

**Scenario A — DB registry only**
- All metadata stored in DB, runtime reads from DB.

**Scenario B — DB + manifest file**
- DB for metadata, plus a generated manifest YAML/JSON used by runtime loaders.

> **Recommended: Scenario B.**

### Feature 4: Local Model Loading on Startup

**What it does:** When your server starts, selected models are already ready.

**Important design decision:**
Do not load every model blindly at startup if you have limited VRAM. Instead create two tiers:
- **Hot models:** loaded on startup
- **Cold models:** loaded on demand

That matters because even small models consume GPU/CPU RAM, and different runtimes handle memory differently.

**Scenario A — Eager loading**
- Load all enabled models at startup.
- Best when: only a few small models, dedicated server, predictable usage

**Scenario B — Warm pool**
- Load only priority models at startup, lazy-load others.
- Best when: laptop GPU, many models, mixed SLM/LLM catalog

> **Recommended: Scenario B.**

### Feature 5: Streaming and Token Feed

**What it does:** Lets your apps receive token-by-token output.

**Must support:**
- SSE streaming
- Cancellation
- Final usage object
- Latency metrics
- Token count estimate or actual usage if backend provides it

**Scenario A — Backend-native stream pass-through**
- If backend supports streaming, forward chunks directly.

**Scenario B — Gateway-managed streaming**
- Gateway reads backend chunks and normalizes format.

> **Recommended: Scenario B**, because your clients get one stable stream format.

### Feature 6: Tool-Calling Compatible Output

You said some models will specialize in tool calling.

**What it does:** Makes outputs reliable for agent orchestration.

**Scenario A — Prompt-enforced JSON**
- Use a strict response schema and validate output.

**Scenario B — Native function/tool calling support**
- Some models and runtimes already expose better support for structured outputs or function-calling-like behavior (e.g., Mistral-7B-Instruct-v0.3 explicitly mentions function calling support).

> **Recommended: Implement both** — native support when available, fallback strict JSON schema validation.

**Also add:**
- Retries on invalid JSON
- Schema validation
- Tool name allowlist
- Max argument size
- Unsafe tool-call rejection

### Feature 7: Local Fine-Tuning Pipeline

**What it does:** Lets you pick base model, dataset, fine-tune config, output name, runtime target. Then your system trains and registers the result.

**Best practical approach:** Start with LoRA/QLoRA, not full fine-tuning. PEFT is specifically built for parameter-efficient fine-tuning, and LoRA reduces the number of trainable parameters dramatically.

**Scenario A — Transformers + PEFT + TRL**
- Best for: full control, custom code, direct Python integration, learning deeply
- TRL's SFTTrainer supports common dataset formats and is a clean starting point for supervised fine-tuning.

**Scenario B — Axolotl orchestrated training**
- Best for: faster setup, config-driven training, easier repeatability, multi-GPU growth
- Axolotl documents LoRA/QLoRA workflows and quickstarts for 1B-scale fine-tuning.

> **Recommended:** Start with Transformers + PEFT + TRL for your custom platform, optionally support Axolotl as an advanced execution backend later.

### Feature 8: Remote Fine-Tuning Execution

This is one of your most important features.

**What it does:** Your local platform becomes the control plane. Training may happen locally, on Colab, on a remote GPU VM, or on a rented provider.

**Architecture:** Your local system should not depend on the training box being "special". Treat remote training as a job runner.

**Scenario A — Remote container job**
- Your local server: packages dataset + config, uploads them, starts a container remotely, polls logs/status, downloads artifacts after completion, registers model locally.
- Best when: using your own remote server, repeatability matters

**Scenario B — Remote agent runner**
- A lightweight Python "worker agent" runs on remote GPU box. Your local server tells it: what to train, where to download data, where to upload results.
- Best when: you may use many providers, you want flexible orchestration

> **Recommended: Scenario B** for your long-term system.

**For Colab specifically:**
Colab is possible, but it is less production-stable than your own VM because sessions can reset and environments are more ephemeral. Treat Colab as a convenience backend, not your primary training infrastructure.

### Feature 9: Remote Inference Bridge

This is also a core requirement.

**What it does:** Your local application remains the single endpoint. It decides whether to serve local or remote model.

**Scenario A — Transparent proxy**
- Local API forwards request to remote runtime and streams result back.
- Best when: remote runtime already exposes OpenAI-like API

**Scenario B — Managed broker**
- Local API keeps routing rules, authentication, health checks, retries, fallback, queueing.
- Best when: you want reliability, you may have multiple remote servers

> **Recommended: Scenario B.**

**Add these controls:**
- Per-model route target
- Health status
- Fallback target
- Timeout
- Max concurrency
- Local backup model

### Feature 10: Import Pre-Fine-Tuned Model ZIPs

**What it does:** Lets you upload models from Colab or any other machine.

**Accept these package types:**
- Full Hugging Face model folder
- LoRA adapter folder
- Merged safetensors model
- GGUF model folder
- Tokenizer bundle
- Manifest JSON

**Import flow:**
1. Upload zip
2. Checksum scan
3. Unzip to quarantine path
4. Validate required files
5. Detect type
6. Attach or choose base model if adapter
7. Register in model registry
8. Optional convert for selected runtime
9. Mark ready

**Scenario A — Strict package contract**
- Require every uploaded zip to include a `model_manifest.json`
- Best for: reliability, automation

**Scenario B — Smart autodetect**
- System infers format by file contents
- Best for: convenience

> **Recommended:** Support both, but prefer manifest.

> **Important note:** Ollama imports GGUF models and adapters through a Modelfile, and adapter imports must match the same base model lineage. That same rule should exist in your platform too.

### Feature 11: Model Conversion Pipeline

You will need this.

**What it does:** Converts trained artifacts into runnable formats.

**Common flows:**
- LoRA adapter → serve directly in vLLM
- LoRA adapter + base → merged HF model
- Merged HF model → GGUF
- GGUF → llama.cpp runtime
- HF model → vLLM runtime

**Scenario A — Keep adapters separate**
- Pros: smaller storage, faster experimentation, easier versioning

**Scenario B — Merge adapters into standalone model**
- Pros: simpler deployment, easier sharing, fewer runtime dependencies

> **Recommended:** Keep adapters separate during experimentation, merge only for deployment targets that need it.

### Feature 12: Dataset Management

**What it does:** Stores and validates your training data.

**Required support:**
- JSONL
- Chat/conversation format
- Instruction format
- Tool-call training format
- Plain text corpus (later)

> TRL supports common dataset formats, including conversational datasets.

**Dataset pipeline:**
- Upload
- Schema validation
- Sample preview
- Tokenization preview
- Duplicate detection
- Train/val split
- Versioning
- Lineage tracking

**Scenario A — Raw dataset storage only**
- Store original file only.

**Scenario B — Raw + processed dataset versions**
- Store original and normalized training-ready copy.

> **Recommended: Scenario B.**

### Feature 13: Security Model

This matters because your local API becomes your personal provider.

**Must include:**
- API key auth
- TLS if accessed over network
- Remote backend credentials vault
- Dataset encryption at rest if sensitive
- Signed artifact import (optional)
- Per-model access rules
- Audit logs

**Scenario A — HTTPS with API tokens**
- Best for: remote HTTP inference/training APIs

**Scenario B — WireGuard/VPN or SSH tunnel**
- Best for: private self-hosted GPU nodes

> **Recommended:** Use both depending on provider.

**Also:**
- Never expose raw SSH credentials to app users
- Store secrets in `.env` only for dev; move to a secret store later

### Feature 14: Observability

**What it does:** Lets you trust the system.

**Metrics to record:**
- Request count
- Latency
- Tokens in/out
- Streaming duration
- Model load time
- GPU memory usage
- Training job duration
- Failed generations
- Invalid JSON/tool-call rate

**Scenario A — Logs only**
- Simple JSON logs.

**Scenario B — Logs + metrics + tracing**
- Use: Prometheus, Grafana, OpenTelemetry (later)

> **Recommended:** Scenario B eventually, but start with structured JSON logs plus Prometheus.

### Feature 15: Versioning and Rollback

**What it does:** Lets you compare and revert models safely.

**Version each:**
- Dataset
- Training config
- Base model
- Adapter
- Merged model
- Prompt template
- Runtime parameters

**Scenario A — Semantic versioning**
- Example: `planner-qwen-3b:v1.2.0`

**Scenario B — Immutable build ids**
- Example: `planner-qwen-3b:2026-03-17-001`

> **Recommended:** Use both.

### Feature 16: Admin UI or CLI

You need at least one control surface.

**Option A — CLI first**
- Best for: speed, developer use, automation

**Option B — Web admin panel**
- Best for: reviewing jobs, dataset previews, model registry editing, logs

> **Recommended:** Start with CLI + API, add UI second.

---

## 5. Suggested Project Structure

```
local_llm_platform/
  apps/
    gateway_api/
    trainer_worker/
    runtime_manager/
    admin_cli/
  core/
    config/
    logging/
    security/
    schemas/
    exceptions/
  services/
    registry/
    artifacts/
    datasets/
    routing/
    streaming/
    auth/
    metrics/
  runtimes/
    base.py
    llama_cpp_runtime.py
    vllm_runtime.py
    tgi_runtime.py
    remote_http_runtime.py
    remote_ssh_runtime.py
  training/
    base.py
    local_trainer.py
    remote_trainer.py
    colab_trainer.py
    import_trainer.py
    pipelines/
      sft_pipeline.py
      lora_pipeline.py
      merge_pipeline.py
      export_gguf_pipeline.py
  models/
    manifests/
    local_store/
    adapters/
    gguf/
  datasets/
    raw/
    processed/
  jobs/
    queue/
    logs/
  db/
    migrations/
  tests/
```

---

## 6. Recommended Implementation Phases

### Phase 1 — Foundation

**Build:**
- FastAPI gateway
- SQLite/Postgres registry
- `/v1/models`
- `/v1/chat/completions`
- API key auth
- One local runtime backend

**Choose backend:**
- Choose **llama.cpp** first if your first target is laptop/local quantized models
- Choose **vLLM** first if your first target is stronger GPU server and transformer-native serving

### Phase 2 — Dynamic Model Registry

**Build:**
- Model registration service
- Hot vs cold model logic
- Health checks
- Startup load policies
- Model capability metadata

### Phase 3 — Streaming and Structured Output

**Build:**
- SSE stream normalization
- JSON schema response mode
- Tool-call validation
- Cancellation support

### Phase 4 — Fine-Tuning Local

**Build:**
- Dataset upload
- Dataset validation
- Training job schema
- Local SFT LoRA pipeline
- Training logs
- Output artifact registration

> Use: Transformers + PEFT + TRL first

### Phase 5 — Import External Models

**Build:**
- Zip upload
- Manifest parser
- Artifact validator
- Runtime compatibility checker
- Registration flow

### Phase 6 — Remote Training Bridge

**Build:**
- Remote worker protocol
- Remote job launch
- Upload/download artifacts
- Logs polling
- Secure auth

### Phase 7 — Remote Inference Bridge

**Build:**
- Remote runtime entries in registry
- Transparent proxy streaming
- Health checks
- Fallback routing

### Phase 8 — Production Features

**Build:**
- Metrics
- Model versioning
- Rollback
- Admin CLI/UI
- Prometheus/Grafana
- Artifact backup
- Concurrency controls

---

## 7. Recommended First Technical Decisions

### Inference
- **Gateway:** FastAPI
- **Small local SLMs:** llama.cpp
- **Transformer server / bigger local or remote:** vLLM
- **API format:** OpenAI-compatible

> This is practical because vLLM already exposes OpenAI-compatible serving and LoRA serving, while llama.cpp is strong for local GGUF workflows.

### Fine-Tuning
- **Local training:** Transformers + PEFT + TRL
- **Advanced/repeatable training later:** Axolotl
- **Main method:** LoRA/QLoRA, not full fine-tuning

> PEFT/LoRA are specifically intended to reduce compute and storage cost, and Axolotl offers config-driven LoRA/QLoRA workflows.

### Registry
- **Postgres** if serious
- **SQLite** if starting now

### Queue
- **Redis + Celery/RQ**

### Remote
- HTTP API for inference
- SSH/WireGuard for trusted self-hosted machines
- Worker agent on remote GPU machines

---

## 8. Base Model Strategy for Specialized Models

You asked about having separate specialized models. A strong practical setup is:

- **Tool-calling model:** small instruct model with strong structured output behavior
- **Personality model:** small instruct model fine-tuned for style/persona
- **Planning/reasoning model:** somewhat larger instruct model
- **Code/tool-execution assistant:** coder-specialized base

**Reasonable current open-weight families:**
- Llama 3.2 1B/3B Instruct
- Qwen2.5 0.5B/1.5B/3B Instruct
- Mistral 7B Instruct

> Meta's Llama 3.2 text models are available in 1B and 3B sizes and are positioned for dialogue, agentic retrieval, and summarization use cases. Qwen2.5 provides models from 0.5B to 72B. Mistral-7B-Instruct-v0.3 explicitly notes function-calling support.
>
> For a laptop-oriented system, the 1B–3B range is the safest place to start.

---

## 9. Risks and Design Limits

You should plan for these early:

### VRAM/RAM Pressure
- **Problem:** Loading many models at startup may not be realistic on a laptop.
- **Solution:** Hot/cold loading, quantization, model eviction policy

### Different Chat Templates
- **Problem:** Different models require different prompt formatting.
- **Solution:** Store chat template per model in registry

### Adapter Compatibility
- **Problem:** LoRA adapters are tied to their base model family.
- **Solution:** Registry must enforce base-model match at import time

### Colab Instability
- **Problem:** Remote training on Colab can disconnect or expire.
- **Solution:** Make Colab a convenience backend, not your main backend

### Structured Output Reliability
- **Problem:** Small models can drift from schema.
- **Solution:** Validator + retry + constrained prompt templates

### Artifact Sprawl
- **Problem:** Fine-tuning creates lots of files.
- **Solution:** Artifact retention rules, version cleanup, immutable manifests

---

## 10. Minimum Viable Version

**Build this first:**

- FastAPI gateway
- Local llama.cpp backend
- Model registry
- `/v1/models`
- `/v1/chat/completions` with streaming
- Model import from folder/zip
- One local LoRA fine-tuning pipeline
- Remote model route entry
- Admin CLI

**That already gives you:**
- Your own local provider
- Importable models
- Trainable specialized SLMs
- One API for your agents

---

## 11. Best Review Checklist

When reviewing the implementation, check these one by one:

1. Can one client call every model through one API?
2. Can local and remote models be switched without changing the client?
3. Can imported Colab models be validated and registered automatically?
4. Can the platform stream tokens reliably?
5. Can it enforce structured tool outputs?
6. Can it fine-tune locally?
7. Can it launch remote training and pull artifacts back?
8. Can it version and roll back models?
9. Can it survive backend changes without breaking your apps?

---

## 12. Recommended Final Direction

### Version 1
- FastAPI gateway
- llama.cpp local serving
- SQLite/Postgres registry
- Zip import
- Local LoRA training
- OpenAI-compatible API
- Streaming
- CLI admin

### Version 2
- vLLM backend
- Remote HTTP inference bridge
- Remote training worker
- Artifact manifest system
- Prometheus metrics

### Version 3
- Axolotl backend option
- Admin web UI
- Multi-node routing
- Automatic model warm pool
- Fallback and load balancing
