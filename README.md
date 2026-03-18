# Local LLM Platform (LLP)

A local-first LLM/SLM platform with an OpenAI-compatible API. Run, fine-tune, and manage local and remote language models through a single unified interface.

## Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API clients
- **Multi-Backend Runtime Support** - llama.cpp, vLLM, TGI, and remote HTTP backends
- **Model Registry** - Automatic model discovery, registration, and metadata management
- **Streaming** - SSE-based token-by-token streaming
- **Fine-Tuning** - Local LoRA/QLoRA fine-tuning with PEFT and TRL
- **Model Import** - Import pre-trained models from Colab or other sources
- **Remote Inference** - Route requests to remote GPU servers
- **Remote Training** - Launch training jobs on remote machines
- **Admin CLI** - Command-line interface for platform management
- **Desktop UI** - Optional GUI for model management

## Quick Start

### Prerequisites

- Python 3.10+
- Redis (for job queue)
- Optional: CUDA-compatible GPU

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/local-llm-platform.git
cd local-llm-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r requirements.txt

# Install with optional extras
pip install -e ".[training]"    # Training support (PEFT, TRL, etc.)
pip install -e ".[runtimes]"    # llama.cpp runtime
pip install -e ".[desktop]"     # Desktop UI
pip install -e ".[all]"         # Everything
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
# Set DATABASE_URL, REDIS_URL, API_KEY, etc.
```

### Running the Platform

```bash
# Start all services (recommended)
python run_platform.py

# Start gateway only
python run_platform.py --gateway-only

# Start without GUI
python run_platform.py --no-gui

# Kill leftover processes
python run_platform.py --kill

# Using CLI
llp serve
```

### Docker

```bash
# Build and start
docker-compose up -d

# Stop
docker-compose down
```

## API Endpoints

| Endpoint                     | Method | Description              |
|------------------------------|--------|--------------------------|
| `/v1/models`                 | GET    | List available models    |
| `/v1/chat/completions`       | POST   | Chat completions         |
| `/v1/completions`            | POST   | Text completions         |
| `/v1/embeddings`             | POST   | Generate embeddings      |
| `/health`                    | GET    | Health check             |
| `/docs`                      | GET    | Interactive API docs     |

### Example Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-local-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"  # or leave empty if auth disabled
)

response = client.chat.completions.create(
    model="my-local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Architecture

```
local_llm_platform/
  apps/
    gateway_api/          # OpenAI-compatible API gateway
    trainer_worker/       # Fine-tuning job worker
    runtime_manager/      # Model runtime lifecycle
    admin_cli/            # CLI administration tool
  core/
    config/               # Configuration management
    logging/              # Structured logging
    security/             # Authentication & authorization
    schemas/              # Pydantic models
    exceptions/           # Custom exceptions
  services/
    registry/             # Model registry service
    artifacts/            # Artifact storage
    datasets/             # Dataset management
    routing/              # Request routing
    streaming/            # SSE streaming
    auth/                 # Authentication
    metrics/              # Observability
  runtimes/
    base.py               # Runtime interface
    llama_cpp_runtime.py  # llama.cpp backend
    vllm_runtime.py       # vLLM backend
    remote_http_runtime.py # Remote HTTP backend
  training/
    base.py               # Trainer interface
    local_trainer.py      # Local training
    remote_trainer.py     # Remote training
    pipelines/            # Training pipelines
  models/                 # Model storage
  datasets/               # Dataset storage
  jobs/                   # Job queue & logs
  db/                     # Database
```

## Services

| Service          | Port | Description                     |
|------------------|------|---------------------------------|
| Gateway API      | 8000 | OpenAI-compatible API           |
| Runtime Manager  | 8001 | Model lifecycle management      |
| Trainer Worker   | 8002 | Fine-tuning job execution       |
| Redis            | 6379 | Job queue and caching           |

## Model Management

### Importing Models

```bash
# Import a GGUF model
llp import path/to/model.gguf --name my-model

# Import from zip (e.g., from Colab)
llp import path/to/model.zip --name my-model

# List models
llp models list

# Remove a model
llp models remove my-model
```

### Model Formats Supported

- **GGUF** - llama.cpp quantized models
- **SafeTensors** - Hugging Face models
- **LoRA Adapters** - PEFT fine-tuned adapters
- **Zip Archives** - Full model packages with manifest

## Fine-Tuning

### Local Training

```bash
# Start a fine-tuning job
llp train \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --dataset datasets/my_dataset.jsonl \
  --output my-fine-tuned-model \
  --method lora \
  --epochs 3
```

### Dataset Format

```jsonl
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help?"}]}
```

## Configuration

Key environment variables (see `.env.example`):

| Variable                 | Default                              | Description            |
|--------------------------|--------------------------------------|------------------------|
| `HOST`                   | `0.0.0.0`                           | Server bind address    |
| `PORT`                   | `8000`                              | Server port            |
| `DATABASE_URL`           | `sqlite:///./local_llm_platform.db` | Database connection    |
| `REDIS_URL`              | `redis://localhost:6379/0`          | Redis connection       |
| `API_KEY`                | (empty)                              | API authentication     |
| `DEFAULT_BACKEND`        | `llama_cpp`                         | Default runtime        |
| `MAX_LOADED_MODELS`      | `3`                                 | Concurrent model limit |
| `TRAINING_WORKER_CONCURRENCY` | `1`                            | Parallel training jobs |

## CLI Commands

```bash
# Start the server
llp serve

# List models
llp models list

# Import model
llp import <path> [--name <name>]

# Start training
llp train --base-model <model> --dataset <data> --output <name>

# View job status
llp jobs list
llp jobs status <job-id>

# System info
llp status
```

## Desktop UI

```bash
# Launch desktop UI
slm-ui

# Or as module
python -m local_llm_platform.apps.desktop.app
```

## Development

```bash
# Install dev dependencies
pip install -e ".[all]"

# Run tests
python -m pytest

# Code formatting
black local_llm_platform/
isort local_llm_platform/

# Type checking
mypy local_llm_platform/
```

## Project Status

Phase 1 (Foundation) - Core gateway, model registry, and API endpoints.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Local inference
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput serving
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [TRL](https://github.com/huggingface/trl) - Transformer reinforcement learning
