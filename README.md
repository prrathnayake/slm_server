# SLM Platform

**Your private AI agent hub.** Run specialized AI agents locally with a single API entrypoint.

## The Vision

Stop sending your data to third-party APIs. SLM Platform is a local-first solution that lets you deploy specialized AI agents using fine-tuned Small Language Models (SLMs) - optimized for specific tasks while running entirely on your hardware.

> **Why SLMs?** Fine-tuned SLMs outperform general-purpose models on specific tasks with a fraction of the compute requirements. A 1.5B parameter model fine-tuned for your use case can outperform GPT-4 on your specific task.

## Key Benefits

- **Privacy First** - All data stays on your machine. No API calls, no data leaving your network.
- **Single Entry Point** - One OpenAI-compatible API for all your agents. Swap agents without changing client code.
- **Task-Specialized** - Fine-tune SLMs for specific domains: coding, reasoning, persona, tool-calling, or custom tasks.
- **Resource Efficient** - Run 3B models on consumer GPUs, 7B models on gaming GPUs. No datacenter required.
- **OpenAI Compatible** - Drop-in replacement for existing apps. Change the base URL, keep your code.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Applications                         │
│        (OpenAI-compatible client, any framework)             │
└─────────────────────┬───────────────────────────────────────┘
                      │ base_url="http://localhost:8000/v1"
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   SLM Platform Gateway                       │
│              (OpenAI-compatible API)                         │
│                                                              │
│  POST /v1/chat/completions  →  Routes to specialized agent  │
│                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  Coder SLM  │ │ Planner SLM │ │  Persona    │  ...      │
│  │   (1.5B)    │ │   (3B)     │ │   (0.5B)    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                              │
│  Each SLM is specialized for one task                        │
└─────────────────────────────────────────────────────────────┘
```

## Use Cases

| Agent Type | Model Size | Use Case |
|------------|------------|----------|
| Code Assistant | 1.5B-3B | Code completion, debugging, refactoring |
| Planner | 3B-7B | Task decomposition, reasoning chains |
| Persona | 0.5B-1.5B | Consistent voice, tone, style |
| Tool Caller | 3B | Function calling, API interactions |
| Custom | Any | Fine-tune with your data |

## Quick Start

### Prerequisites

- Python 3.10+
- Redis (for job queue)
- Optional: NVIDIA GPU (CUDA 11.8+)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/slm-platform.git
cd slm-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r requirements.txt

# Install with extras
pip install -e ".[training]"    # Fine-tuning support
pip install -e ".[runtimes]"    # llama.cpp runtime
pip install -e ".[desktop]"     # Desktop UI
pip install -e ".[all]"         # Everything
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your settings
```

### Running

```bash
# Start the platform
python run_platform.py

# Or start services individually
llp serve
```

### Your First Agent

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local"  # Any value works locally
)

# Use a specialized agent
response = client.chat.completions.create(
    model="coder-qwen-1.5b",  # Your fine-tuned coder agent
    messages=[
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "Write a fast fibonacci function"}
    ]
)
print(response.choices[0].message.content)
```

### Testing with cURL

```bash
# Start the platform first
python -m uvicorn local_llm_platform.apps.gateway_api.main:app --host 0.0.0.0 --port 8000

# Chat completions (non-streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slm-ft-final-merged",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'

# Chat completions (streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slm-ft-final-merged",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "max_tokens": 256
  }'

# Text completions
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slm-ft-final-merged",
    "prompt": "Once upon a time",
    "max_tokens": 100
  }'

# List available models
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health
```

## Fine-Tuning: Create Your Agents

### Why Fine-Tune?

A general-purpose model is mediocre at everything. Fine-tune it once for your task:

```
Base Model (Qwen 1.5B)
     │
     ▼
┌─────────────────┐
│ Your Dataset    │  Example: 500 examples of your coding style
│ (100-1000 rows)│
└────────┬────────┘
         │
         ▼ Fine-tune with LoRA (2-4 hours on RTX 3060)
              │
              ▼
     ┌──────────────────┐
     │ Your Coder SLM   │  Better than GPT-4 at YOUR codebase
     │ (task-specialized)│
     └──────────────────┘
```

### Create a Specialized Agent

1. **Prepare your training data** (JSONL format):

```json
{"messages": [
  {"role": "system", "content": "You are a security expert."},
  {"role": "user", "content": "How to prevent SQL injection?"},
  {"role": "assistant", "content": "Use parameterized queries..."}
]}
```

2. **Fine-tune**:

```bash
llp train \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset my-security-data.jsonl \
  --output security-expert \
  --method lora
```

3. **Use it**:

```python
response = client.chat.completions.create(
    model="security-expert",
    messages=[{"role": "user", "content": "Fix this SQL: 'SELECT * FROM users WHERE id=' + user_id'"}]
)
```

## Architecture

```
slm_platform/
  apps/
    gateway_api/          # OpenAI-compatible API gateway
    trainer_worker/       # Fine-tuning worker
    runtime_manager/      # Model lifecycle management
    admin_cli/            # CLI tool
    desktop/              # Desktop UI
  core/
    config/              # Configuration
    logging/              # Structured logging
    security/             # Authentication
    schemas/              # Pydantic models
  services/
    registry/             # Model registry
    artifacts/            # Artifact storage
    datasets/             # Dataset management
    routing/              # Request routing
    streaming/            # SSE streaming
  runtimes/
    llama_cpp/            # GGUF model runtime
    vllm/                 # vLLM runtime
    transformers/         # HuggingFace runtime
    remote/               # Remote inference
  training/
    pipelines/            # LoRA/QLoRA/SFT pipelines
  models/                 # Local model storage
  datasets/               # Training data
```

## Services

| Service          | Port | Description                     |
|------------------|------|---------------------------------|
| Gateway API      | 8000 | OpenAI-compatible API           |
| Runtime Manager  | 8001 | Model lifecycle                |
| Trainer Worker   | 8002 | Fine-tuning execution          |
| Redis            | 6379 | Job queue                      |

## API Endpoints

| Endpoint                 | Method | Description              |
|--------------------------|--------|-------------------------|
| `/v1/chat/completions`    | POST   | Chat with agents        |
| `/v1/completions`        | POST   | Text completion         |
| `/v1/models`             | GET    | List available agents   |
| `/health`                | GET    | Health check           |
| `/docs`                  | GET    | Interactive docs        |

## CLI

```bash
llp serve                    # Start the platform
llp models list             # List registered agents
llp models load <name>      # Load an agent into memory
llp train --help            # Fine-tune a new agent
llp datasets upload <file>  # Upload training data
llp status                  # System status
```

## Desktop UI

Launch the GUI for visual management:

```bash
slm-ui
```

Features:
- Dashboard with system overview
- Model management (download, register, load/unload)
- Training job monitoring
- Dataset upload and preview
- Import models from zip files
- **Help & Docs** for guided tutorials

## Model Formats

| Format       | Best For                  | Runtime        |
|--------------|---------------------------|----------------|
| GGUF (Q4/Q5) | CPU inference, laptops    | llama.cpp      |
| SafeTensors  | GPU inference, fine-tuning| vLLM, HF       |
| LoRA Adapter | Task-specific fine-tuning | Any base model |

## Privacy & Security

- **100% Local** - No network calls to external APIs
- **Your Data** - Conversations never leave your machine
- **API Key** - Optional authentication for remote access
- **TLS** - Enable HTTPS for network deployment

## Development

```bash
# Install all dependencies
pip install -e ".[all]"

# Run tests
python -m pytest

# Code style
black local_llm_platform/
ruff check local_llm_platform/
```

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient local inference
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput serving
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [TRL](https://github.com/huggingface/trl) - Transformer training
- [HuggingFace](https://huggingface.co/) - Model hub
