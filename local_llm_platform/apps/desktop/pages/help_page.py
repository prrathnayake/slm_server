import customtkinter as ctk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


HELP_SECTIONS = {
    "quick_start": {
        "title": "Welcome to SLM Platform",
        "icon": "\u2665",
        "content": """PRIVACY-FIRST AI AGENTS - ALL LOCALLY RUN

SLM Platform lets you run specialized AI agents locally.
No data leaves your machine. No API fees. Full control.

==============================================
QUICK START (5 minutes)
==============================================

STEP 1: Start the Platform
  - Make sure the server is running:
    python run_platform.py
  - Check the sidebar - green dot = connected

STEP 2: Download a Base Model
  - Go to Models page
  - Click "Download Base"
  - Enter: Qwen/Qwen2.5-1.5B-Instruct
  - Wait for download (1-3GB)

STEP 3: Fine-Tune for Your Task
  - Go to Datasets page
  - Upload examples of your task
  - Go to Training page
  - Select your model and dataset
  - Click "Start Training"

STEP 4: Use Your Agent
  - Load your fine-tuned model
  - Use any OpenAI-compatible app
  - Point to: http://localhost:8000/v1

==============================================
WHY SLMs?
==============================================

Small Language Models (SLMs) fine-tuned for
specific tasks outperform general models:

  - FASTER: 10-50x less compute
  - CHEAPER: Run on your laptop/PC
  - BETTER: Exceeds GPT-4 on specific tasks
  - PRIVATE: Everything runs locally

Example: A 1.5B model fine-tuned on 500
examples of your code style will write code
in YOUR style better than any general model.

==============================================
NEXT STEPS
==============================================

- Read "Creating Specialized Agents"
- Explore "Fine-Tuning Guide"
- Check "Using the API" for integration
- See "Privacy & Security" for details""",
    },
    "agents": {
        "title": "Creating Specialized Agents",
        "icon": "\u2B21",
        "content": """TURN BASE MODELS INTO SPECIALIZED AGENTS

A general model is mediocre at everything.
Fine-tune once, be exceptional at one thing.

==============================================
WHAT IS AN AGENT?
==============================================

An "agent" is a specialized SLM optimized
for a specific task or personality:

  - CODING AGENT: Your codebase expert
  - PLANNING AGENT: Task decomposer
  - PERSONA AGENT: Consistent voice/style
  - TOOL AGENT: Function calling expert
  - CUSTOM: Anything you can define

==============================================
HOW TO CREATE AN AGENT
==============================================

1. DEFINE YOUR TASK
   What should this agent excel at?
   Examples:
   - "Review Python code for bugs"
   - "Write in Shakespeare's style"
   - "Explain blockchain concepts simply"
   - "Draft professional emails"

2. GATHER EXAMPLES (100-1000)
   Collect examples of ideal responses
   Format: JSONL (see Datasets page help)
   
   Example for coding agent:
   {"messages": [
     {"role": "user", "content": "Fix this bug..."},
     {"role": "assistant", "content": "The issue is..."}
   ]}

3. FINE-TUNE THE MODEL
   - Go to Training page
   - Select base model (1.5B-3B recommended)
   - Select your dataset
   - Choose LoRA (recommended)
   - Set output name: "my-coding-agent"
   - Start training

4. REGISTER & LOAD
   - Training auto-registers the model
   - Go to Models page
   - Click "Load" on your new agent

5. USE IT
   Change the model name in your app:
   
   client.chat.completions.create(
       model="my-coding-agent",
       messages=[...]
   )

==============================================
RECOMMENDED SETUP
==============================================

Hardware:                    Agent Type:
- RTX 3060+ (8GB):          Coding, Tool-calling
- RTX 4060+ (12GB):         Planning, Complex reasoning
- Laptop (no GPU):          Lightweight persona agents
- MacBook M1+:              All types (Metal support)

Base Models:
- Qwen/Qwen2.5-1.5B-Instruct (fast, capable)
- Qwen/Qwen2.5-3B-Instruct (balanced)
- meta-llama/Llama-3.2-1B-Instruct
- microsoft/Phi-3-mini-4k-instruct

==============================================
TIPS FOR BETTER AGENTS
==============================================

1. QUALITY > QUANTITY
   100 great examples > 1000 mediocre ones

2. CONSISTENT FORMAT
   Keep your JSONL format consistent

3. DIVERSE EXAMPLES
   Include edge cases and variations

4. SYSTEM PROMPTS
   Fine-tune captures style, use system
   prompts for task-specific instructions""",
    },
    "import_models": {
        "title": "Importing Models",
        "icon": "\u2913",
        "content": """IMPORT MODELS FROM ANYWHERE

Bring in models trained elsewhere - from Google
Colab, other machines, or online sources.

==============================================
SUPPORTED FORMATS
==============================================

  GGUF (.gguf files)
    - Quantized for efficient inference
    - Best for llama.cpp runtime
    - Works on CPU or GPU

  SafeTensors (.safetensors)
    - Full precision model weights
    - Best for fine-tuning
    - Requires GPU

  LoRA Adapters (.bin, .pt)
    - Fine-tuned weights overlay
    - Small file size
    - Must attach to base model

  Full Model Folders (ZIP)
    - Complete model packages
    - Include config, tokenizer
    - Best for import

==============================================
IMPORT FROM GOOGLE COLAB
==============================================

1. In Colab, after training:
   import shutil
   shutil.make_archive('my_agent', 'zip', 'output_folder')

2. Download the ZIP file

3. In this app:
   - Go to Import page
   - Click "Browse" -> select your ZIP
   - Click "Import Model"

4. Done! Find it in Models page

==============================================
IMPORT GGUF FILES
==============================================

1. Download .gguf file from HuggingFace
   Example: TheBloke/Llama-2-7B-Chat-GGUF

2. Place in: models/gguf/

3. Register manually:
   - Go to Models page
   - Click "+ Register"
   - Backend: llama_cpp
   - Format: gguf
   - Path: models/gguf/your-model.gguf

==============================================
IMPORT WITH MANIFEST
==============================================

For best results, include model_manifest.json:

{
  "model_id": "my-agent",
  "display_name": "My Coding Agent",
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "runtime_backend": "transformers",
  "model_format": "adapter",
  "specialization": "coding"
}

Place manifest.json in the same folder
as your model files before importing.

==============================================
VALIDATION
==============================================

The import process:
1. Extracts files to quarantine folder
2. Validates required files exist
3. Detects model format automatically
4. Registers in the model database
5. Moves to appropriate storage

Check Import Log for warnings/errors.""",
    },
    "training": {
        "title": "Fine-Tuning Guide",
        "icon": "\u2699",
        "content": """FINE-TUNE SLMs FOR YOUR SPECIFIC TASK

Fine-tuning transforms a general model into
a specialized agent that excels at your task.

==============================================
WHY FINE-TUNE?
==============================================

Pre-trained models know everything generally.
Fine-tuning makes them experts at one thing.

  WITHOUT FINE-TUNING:
  - Generic responses
  - Inconsistent style
  - Misses domain specifics

  WITH FINE-TUNING:
  - Expert-level output
  - YOUR style/preferences
  - Domain terminology

==============================================
TRAINING TYPES
==============================================

  LoRA (Recommended)
    - Trains only 0.1-1% of parameters
    - Fast training (1-4 hours)
    - Low VRAM requirement (6-8GB)
    - Good for most use cases

  QLoRA
    - Quantized base + LoRA
    - Even lower memory
    - Slower than LoRA
    - Good for 7B+ models

  SFT (Full Fine-tune)
    - Trains all parameters
    - Requires 24GB+ VRAM
    - Best quality but expensive
    - For research/production

==============================================
STEP-BY-STEP TRAINING
==============================================

1. PREPARE DATA
   - Go to Datasets page
   - Upload JSONL file
   - Use consistent format

2. CONFIGURE TRAINING
   - Base Model: Qwen/Qwen2.5-1.5B-Instruct
   - Dataset: your-uploaded-dataset
   - Output Name: my-specialized-agent
   - Type: lora (recommended)

3. ADJUST PARAMETERS
   - Epochs: 2-3 (start small)
   - Batch Size: 4 (reduce if OOM)
   - Learning Rate: 2e-4 (good default)
   - LoRA Rank (r): 16 (higher = more capacity)
   - Max Seq Length: 2048 (adjust to data)

4. START TRAINING
   - Click "Start Training"
   - Monitor progress bar
   - Check logs for loss values

5. LOAD & USE
   - Training auto-registers model
   - Go to Models page
   - Load your new agent

==============================================
MONITORING TRAINING
==============================================

Watch these indicators:

  Progress Bar: Completion percentage
  Current Epoch: Which epoch (of total)
  Loss: Should decrease over time
  
  Good signs:
  - Loss decreasing steadily
  - No sudden spikes
  - Epochs completing

  Warning signs:
  - Loss not decreasing after epoch 1
  - Loss = NaN (overflow, reduce LR)
  - Very slow progress (reduce batch size)

==============================================
DATASET FORMATS
==============================================

JSONL (Recommended):
{"messages": [
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "User question..."},
  {"role": "assistant", "content": "Assistant answer..."}
]}

Instruction Format:
{"instruction": "...", "output": "..."}

Chat Format:
{"conversations": [
  {"from": "human", "value": "..."},
  {"from": "gpt", "value": "..."}
]}

==============================================
TROUBLESHOOTING
==============================================

OUT OF MEMORY:
  - Reduce batch_size to 2
  - Reduce max_seq_length to 1024
  - Use QLoRA instead of LoRA
  - Use smaller base model

TRAINING TOO SLOW:
  - Use smaller model (1.5B not 7B)
  - Reduce max_seq_length
  - Use GGML quantized base

BAD RESULTS:
  - More training data (100+ examples)
  - Better data quality
  - Try different learning rate
  - More epochs (if loss still dropping)""",
    },
    "models": {
        "title": "Managing Models",
        "icon": "\u2B21",
        "content": """MODEL MANAGEMENT

Register, load, unload, and organize your
agents and base models.

==============================================
MODEL TABLE COLUMNS
==============================================

  Model ID:      Unique identifier
  Display Name:  Human-readable name
  Format:        gguf, safetensors, adapter
  Backend:       Which runtime handles it
  Status:        Current state

STATUS INDICATORS:
  green  = Ready (loaded in memory)
  orange = Loading (being loaded)
  gray   = Unloaded (on disk)
  blue   = Training (being fine-tuned)
  red    = Failed (error occurred)

==============================================
MODEL ACTIONS
==============================================

  [Load]     - Load into memory (enable use)
  [Unload]   - Free from memory (save RAM)
  [Info]     - View adapter details (LoRA)
  [Merge]    - Combine LoRA with base model
  [X]        - Remove from registry (not files)

==============================================
DOWNLOAD BASE MODELS
==============================================

Recommended starter models:

  Ultra-Light (no GPU):
  - Qwen/Qwen2.5-0.5B-Instruct

  Light (6GB VRAM):
  - Qwen/Qwen2.5-1.5B-Instruct
  - Llama-3.2-1B-Instruct

  Medium (8GB VRAM):
  - Qwen/Qwen2.5-3B-Instruct
  - Llama-3.2-3B-Instruct

  Strong (12GB+ VRAM):
  - Qwen/Qwen2.5-7B-Instruct
  - Mistral-7B-Instruct-v0.3

==============================================
REGISTERING MODELS
==============================================

For models on disk:

1. Click "+ Register"

2. Fill details:
   Model ID:     my-agent
   Display Name: My Coding Agent
   Backend:      transformers (or llama_cpp)
   Format:       safetensors or gguf
   Specialization: coding
   Artifact Path: /full/path/to/model

3. Click "Register"

==============================================
ADAPTERS (LoRA)
==============================================

LoRA adapters are lightweight overlays:

  [A] Prefix = Adapter in the list
  
  Info button: View adapter details
    - Base model it attaches to
    - LoRA rank and alpha
    - Target modules

  Merge button: Create standalone model
    - Combines adapter + base
    - Creates new .safetensors file
    - Register as new model

==============================================
MEMORY MANAGEMENT
==============================================

Loaded models consume RAM/VRAM:

  1.5B model: ~3GB RAM
  3B model:   ~6GB RAM
  7B model:   ~14GB RAM

Tips:
  - Unload unused models
  - Limit MAX_LOADED_MODELS in settings
  - Use quantized (GGUF Q4/Q5) models""",
    },
    "api_usage": {
        "title": "Using the API",
        "icon": "\u2139",
        "content": """OPENAI-COMPATIBLE API

All your existing OpenAI code works with
SLM Platform. Just change the base URL.

==============================================
BASIC USAGE
==============================================

Python (openai library):
------------------------
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local"  # Any value works
)

response = client.chat.completions.create(
    model="my-coding-agent",
    messages=[
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "Write a fast sort"}
    ]
)
print(response.choices[0].message.content)

curl:
-----
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "my-coding-agent",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

==============================================
STREAMING
==============================================

For real-time token-by-token output:

client.chat.completions.create(
    model="my-agent",
    messages=[...],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

==============================================
AVAILABLE ENDPOINTS
==============================================

POST /v1/chat/completions
  - Chat interface (recommended)
  - Use for conversations

POST /v1/completions
  - Text completion (raw)
  - Use for autocomplete

GET /v1/models
  - List all available models

GET /health
  - Check server status

GET /docs
  - Interactive API documentation

==============================================
PARAMETERS
==============================================

  model:        Agent/model ID to use
  messages:     Conversation history
  temperature: Creativity (0.0-2.0)
                0.0 = deterministic
                1.0 = creative (default)
  max_tokens:  Max response length
  top_p:       Nucleus sampling
  stream:      Enable streaming
  stop:        Stop sequences

==============================================
MULTI-AGENT ROUTING
==============================================

Route to different agents by model name:

# Coding agent
client.chat.completions.create(
    model="coder-qwen-1.5b",
    messages=[{"role": "user", "content": "Fix this bug..."}]
)

# Planning agent
client.chat.completions.create(
    model="planner-3b",
    messages=[{"role": "user", "content": "Break down this task..."}]
)

# Persona agent
client.chat.completions.create(
    model="shakespeare-0.5b",
    messages=[{"role": "user", "content": "Say hello..."}]
)""",
    },
    "privacy": {
        "title": "Privacy & Security",
        "icon": "\u26BF",
        "content": """YOUR DATA NEVER LEAVES YOUR MACHINE

SLM Platform is designed for privacy-first
AI. Everything runs locally.

==============================================
HOW WE PROTECT PRIVACY
==============================================

100% LOCAL PROCESSING
  - Models run on YOUR hardware
  - No network requests to external APIs
  - Conversations stored only on your disk

NO TELEMETRY
  - No analytics or tracking
  - No usage data collection
  - No phone-home features

OPTIONAL REMOTE ACCESS
  - Enable TLS for network access
  - API key authentication available
  - VPN/SSH tunnel recommended

==============================================
DATA FLOW
==============================================

WITHOUT SLM PLATFORM:
  User Query
      |
      v
  Third-Party API (OpenAI, Anthropic, etc.)
      |
      v
  Your data on THEIR servers <-- Privacy risk
      |
      v
  Response

WITH SLM PLATFORM:
  User Query
      |
      v
  Your Local SLM Platform <-- Data stays here
      |
      v
  Local Model (GGUF/SafeTensors)
      |
      v
  Response

==============================================
SECURITY BEST PRACTICES
==============================================

1. API KEY
   - Set API_KEY in .env for remote access
   - Use strong, random keys
   - Rotate periodically

2. NETWORK ACCESS
   - Use VPN for remote access
   - Enable TLS/HTTPS in production
   - Firewall: only allow trusted IPs

3. MODEL SOURCES
   - Download from trusted sources
   - Verify checksums when available
   - Use HuggingFace for verified models

4. DOCKER
   - Keep Docker updated
   - Don't expose ports unnecessarily
   - Use read-only volumes for data

==============================================
WHEN TO USE REMOTE APIs
==============================================

Consider cloud APIs for:
  - Very large models (70B+)
  - Specialized models not locally viable
  - Research/experimentation
  - When privacy isn't critical

Use SLM Platform for:
  - Production applications
  - Sensitive data (medical, legal, financial)
  - Cost-sensitive deployments
  - Low-latency requirements
  - Regulatory compliance (GDPR, HIPAA)""",
    },
    "troubleshooting": {
        "title": "Troubleshooting",
        "icon": "\u26A0",
        "content": """COMMON ISSUES AND FIXES

==============================================
SERVER ISSUES
==============================================

"Not Connected" indicator (red dot)
  1. Start server: python run_platform.py
  2. Check server URL in Settings
  3. Ensure port 8000 is available
  4. Check firewall rules

Port already in use
  1. Run: python run_platform.py --kill
  2. Or manually kill process on port

==============================================
MODEL ISSUES
==============================================

Model won't load
  1. Verify model files exist at path
  2. Check correct backend selected
  3. Ensure enough memory available
  4. Check Activity Log for errors

Out of memory (CUDA OOM or RAM)
  1. Unload other models first
  2. Use smaller/quantized model
  3. Training: reduce batch_size
  4. Training: reduce max_seq_length

Model downloads fail
  1. Check internet connection
  2. Verify HuggingFace model exists
  3. Check disk space (1-5GB needed)
  4. Try again (transient issues)

==============================================
TRAINING ISSUES
==============================================

Training fails immediately
  1. Check dataset format is valid
  2. Verify dataset has 100+ examples
  3. Ensure base model is downloaded
  4. Check training logs for errors

Loss = NaN
  1. Reduce learning rate (try 1e-4)
  2. Increase data quality
  3. Check for malformed JSONL

Training too slow
  1. Use smaller base model
  2. Reduce batch_size
  3. Use LoRA instead of SFT
  4. Ensure GPU is being used

==============================================
IMPORT ISSUES
==============================================

ZIP import fails
  1. Ensure ZIP is valid archive
  2. Check for required files inside
  3. Include model_manifest.json
  4. Verify disk space available

Format detection fails
  1. Include manifest.json with files
  2. Check files are in correct structure
  3. For GGUF: ensure .gguf extension

==============================================
PERFORMANCE ISSUES
==============================================

Slow responses
  1. Use smaller model
  2. Use GGUF quantized models
  3. Reduce max_tokens
  4. Close other GPU applications
  5. Ensure using GPU not CPU

GPU not being used
  1. Check CUDA is installed
  2. Verify GPU is detected
  3. Use CUDA_VISIBLE_DEVICES
  4. Check model format (GGUF uses CPU)

==============================================
GET HELP
==============================================

1. Check Activity Log on each page
2. Review logs/ folder for errors
3. API docs: http://localhost:8000/docs
4. GitHub Issues with:
   - Error messages
   - Steps to reproduce
   - Your setup (OS, GPU, Python version)""",
    },
}


class HelpPage(ctk.CTkFrame):
    """Help and documentation page with user-friendly guides."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self._current_section = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.nav_frame = ctk.CTkFrame(self, width=200)
        self.nav_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        self.nav_frame.grid_rowconfigure(99, weight=1)

        ctk.CTkLabel(
            self.nav_frame,
            text="Help & Docs",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w")

        self._nav_buttons = {}
        for i, (section_id, section) in enumerate(HELP_SECTIONS.items()):
            btn = ctk.CTkButton(
                self.nav_frame,
                text=f"  {section['icon']}  {section['title']}",
                anchor="w",
                height=36,
                font=ctk.CTkFont(size=13),
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                command=lambda s=section_id: self._show_section(s),
            )
            btn.grid(row=i + 1, column=0, padx=8, pady=2, sticky="ew")
            self._nav_buttons[section_id] = btn

        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)

        self.header_label = ctk.CTkLabel(
            self.content_frame,
            text="",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.header_label.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")

        self.content_text = ctk.CTkTextbox(
            self.content_frame,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
        )
        self.content_text.grid(row=1, column=0, padx=15, pady=(5, 15), sticky="nsew")

        self._show_section("quick_start")

    def _show_section(self, section_id: str):
        section = HELP_SECTIONS.get(section_id)
        if not section:
            return

        for sid, btn in self._nav_buttons.items():
            if sid == section_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

        self._current_section = section_id

        self.header_label.configure(text=f"{section['icon']}  {section['title']}")

        self.content_text.delete("1.0", "end")
        self.content_text.insert("1.0", section["content"])
        self.content_text.configure(state="disabled")

    def refresh(self):
        pass
