import customtkinter as ctk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


HELP_SECTIONS = {
    "quick_start": {
        "title": "Quick Start Guide",
        "icon": "\u26a1",
        "content": """Welcome to SLM Platform!

This guide will help you get started with running local AI models.

STEP 1: Check Server Connection
  - Look at the bottom-left corner of the sidebar
  - Green dot means connected, red dot means disconnected
  - If disconnected, start the server: python run_platform.py
  - Or update the server URL in Settings

STEP 2: Download or Import a Model
  - Go to Models page and click "Download Base"
  - Or go to Import page to import a model ZIP file
  - Popular starter models:
    > Qwen/Qwen2.5-1.5B-Instruct (fast, lightweight)
    > meta-llama/Llama-3.2-1B-Instruct (good all-rounder)
    > microsoft/Phi-3-mini-4k-instruct (strong reasoning)

STEP 3: Load the Model
  - Find your model in the Models list
  - Click "Load" to make it ready for inference
  - Wait for the status to change to "Ready"

STEP 4: Use the Model
  - The model is now available via the API at:
    http://localhost:8000/v1/chat/completions
  - Use any OpenAI-compatible client or your own apps

Need help? Each page has a Help button with specific instructions.""",
    },
    "import_models": {
        "title": "How to Import Models",
        "icon": "\u2913",
        "content": """Import Page lets you bring models from other sources
into the platform, including models trained on Google
Colab, other machines, or downloaded from the internet.

SUPPORTED FORMATS:
  - GGUF files (quantized models for llama.cpp)
  - SafeTensors (HuggingFace model format)
  - LoRA Adapters (PEFT fine-tuned weights)
  - Full model folders (all files in a directory)
  - ZIP archives (packaged model bundles)

HOW TO IMPORT A MODEL:

1. Go to the Import page from the sidebar

2. Click "Browse" to select a ZIP file
   - The ZIP should contain model files
   - Include a model_manifest.json for best results

3. Optionally enter a Model ID
   - This is a unique name for your model
   - Leave blank to auto-detect from the ZIP

4. Click "Import Model"
   - The system will validate the files
   - Check the Import Log for progress

5. After import, find your model in the Models page
   - Click "Load" to make it ready

ZIP STRUCTURE EXAMPLE:
  my_model.zip
  +-- model.safetensors      (model weights)
  +-- config.json             (model configuration)
  +-- tokenizer.json          (tokenizer files)
  +-- model_manifest.json     (optional metadata)

IMPORTING FROM GOOGLE COLAB:
1. In Colab, save your model to a ZIP:
   !zip -r model.zip model_output/
2. Download the ZIP file
3. Use this Import page to upload it
4. The platform will detect the model type
   and register it automatically

IMPORTING GGUF MODELS:
1. Download a .gguf file from HuggingFace
2. Place it in models/gguf/ folder
3. Go to Models page and click "+ Register"
4. Select backend: llama_cpp
5. Select format: gguf
6. Enter the path to your .gguf file

TIPS:
  - Large models may take several minutes to import
  - Check the Import Log for any warnings
  - Adapters need a matching base model""",
    },
    "training": {
        "title": "How to Train Models",
        "icon": "\u2699",
        "content": """The Training page lets you fine-tune AI models on
your own data to make them better at specific tasks.

WHAT IS FINE-TUNING?
Fine-tuning takes an existing model and trains it
further on your custom data. This makes the model:
  - Better at your specific use case
  - Following your preferred style/format
  - Understanding your domain knowledge

TRAINING TYPES:
  - LoRA: Lightweight adapter training (recommended)
  - QLoRA: Memory-efficient LoRA with quantization
  - SFT: Full supervised fine-tuning

BEFORE YOU START:
1. Prepare your dataset (see Datasets page)
2. Choose a base model (must be downloaded first)
3. Decide on training parameters

HOW TO START TRAINING:

1. Go to the Training page

2. Fill in the configuration:
   Base Model: The model to fine-tune
     Example: Qwen/Qwen2.5-1.5B-Instruct

   Dataset ID: Your uploaded dataset name
     Example: my-training-data

   Output Name: Name for your trained model
     Example: my-custom-model-v1

   Type: Choose lora (recommended for most cases)

3. Set training parameters:
   Epochs: How many times to train on the data
     Default: 3 (good starting point)

   Batch Size: Samples per training step
     Default: 4 (lower if out of memory)

   Learning Rate: How fast the model learns
     Default: 2e-4 (good for LoRA)

   LoRA Rank (r): Adapter complexity
     Default: 16 (higher = more capacity)

   LoRA Alpha: Scaling factor
     Default: 32 (usually 2x the rank)

   Max Seq Length: Maximum text length
     Default: 2048 tokens

4. Choose execution target:
   - local: Train on this machine
   - remote: Train on a remote GPU server

5. Click "Start Training"
   - Monitor progress in the right panel
   - Check logs for detailed information

6. When training completes:
   - The model is automatically registered
   - Find it in the Models page
   - Load and use it like any other model

TRAINING TIPS:
  - Start with small epochs (2-3) to test
  - Use LoRA for most use cases (faster, less memory)
  - Monitor the loss value - it should decrease
  - If loss stops decreasing, training is done
  - If out of memory, reduce batch_size or seq_length

DATASET FORMAT (JSONL):
Each line should be a JSON object with messages:
{"messages": [
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there!"}
]}

MONITORING TRAINING:
  - Progress bar shows completion percentage
  - Current epoch and loss are displayed
  - Logs show detailed training information
  - You can cancel a running job anytime""",
    },
    "models": {
        "title": "Managing Models",
        "icon": "\u2b21",
        "content": """The Models page is your central hub for managing
all AI models in the platform.

MODEL TABLE:
Each model shows:
  - Model ID: Unique identifier
  - Display Name: Human-readable name
  - Format: gguf, safetensors, or adapter
  - Backend: Which runtime handles it
  - Status: ready, loading, unloaded, etc.

MODEL ACTIONS:
  [Load] - Load model into memory for inference
           Status changes to "ready" when done

  [Unload] - Free model from memory
             Use this to save resources

  [X] - Delete model from registry
        Does not delete the actual files

ADAPTERS (LoRA models):
  - Marked with [A] prefix in the list
  - Adapters need a base model to work
  - [Info] - View adapter details
  - [Merge] - Combine adapter with base model
             Creates a standalone model

DOWNLOADING MODELS:
1. Click "Download Base" button
2. Enter a HuggingFace model name:
   - Qwen/Qwen2.5-1.5B-Instruct
   - meta-llama/Llama-3.2-1B-Instruct
   - microsoft/Phi-3-mini-4k-instruct
3. Wait for download to complete
4. The model appears in the list

REGISTERING MODELS:
If you have a model file already on disk:
1. Click "+ Register" button
2. Fill in the details:
   - Model ID: Unique name
   - Display Name: Friendly name
   - Backend: llama_cpp, transformers, etc.
   - Format: gguf, safetensors, adapter
   - Specialization: general, coder, etc.
   - Artifact Path: Path to model files
3. Click "Register"

HOT VS COLD MODELS:
  - Hot models: Loaded at startup (always ready)
  - Cold models: Loaded on demand (saves memory)
  - Configure max loaded models in Settings

POPULAR MODELS:
  Small & Fast (good for laptops):
    - Qwen/Qwen2.5-0.5B-Instruct
    - Qwen/Qwen2.5-1.5B-Instruct
    - microsoft/Phi-3-mini-4k-instruct

  Balanced (needs 4-8GB VRAM):
    - Qwen/Qwen2.5-3B-Instruct
    - meta-llama/Llama-3.2-3B-Instruct

  Powerful (needs 8GB+ VRAM):
    - Qwen/Qwen2.5-7B-Instruct
    - mistralai/Mistral-7B-Instruct-v0.3""",
    },
    "datasets": {
        "title": "Managing Datasets",
        "icon": "\u2637",
        "content": """The Datasets page lets you upload and manage
training data for fine-tuning models.

SUPPORTED FORMATS:
  - JSONL: JSON Lines (recommended)
  - Chat: Conversation format
  - Instruction: Input/output pairs
  - Tool Call: Function calling format
  - Plain Text: Simple text corpus

HOW TO UPLOAD A DATASET:

1. Go to the Datasets page

2. In the Quick Upload section:
   - Click "..." to browse for your file
   - Enter a unique Name for the dataset
   - Select the Format from the dropdown

3. Click "Upload"
   - The dataset appears in the list above

DATASET FORMATS:

JSONL Format (recommended):
  Each line is a JSON object:
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}

Chat Format:
  Conversational data with turns:
  {"conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]}

Instruction Format:
  Simple input-output pairs:
  {"instruction": "...", "input": "...", "output": "..."}

Tool Call Format:
  For training tool-use models:
  {"messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "..."}
  ]}

PREPARING YOUR DATA:
  1. Gather your examples (100+ recommended)
  2. Format them consistently
  3. Save as .jsonl file
  4. Upload via this page
  5. Use the dataset name in Training page

DATA QUALITY TIPS:
  - More diverse examples = better results
  - Consistent formatting is important
  - Remove duplicates and low-quality data
  - Include edge cases in your examples
  - Balance different types of examples
  - 500-2000 examples is a good starting point

SAMPLE DATASET:
  {"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."}
  ]}
  {"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I read a file?"},
    {"role": "assistant", "content": "Use open() function."}
  ]}""",
    },
    "api_usage": {
        "title": "Using the API",
        "icon": "\u2139",
        "content": """The platform exposes an OpenAI-compatible API
that works with any OpenAI client library.

API ENDPOINTS:
  GET  /v1/models              List all models
  POST /v1/chat/completions    Chat with a model
  POST /v1/completions         Text completion
  POST /v1/embeddings          Generate embeddings
  GET  /health                 Check server status

BASE URL: http://localhost:8000

USING WITH PYTHON (openai library):
  from openai import OpenAI

  client = OpenAI(
      base_url="http://localhost:8000/v1",
      api_key="any-value"
  )

  response = client.chat.completions.create(
      model="my-model",
      messages=[
          {"role": "user", "content": "Hello!"}
      ]
  )
  print(response.choices[0].message.content)

USING WITH CURL:
  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
      "model": "my-model",
      "messages": [{"role": "user", "content": "Hello!"}]
    }'

STREAMING:
  Add "stream": true to get token-by-token output:

  stream = client.chat.completions.create(
      model="my-model",
      messages=[{"role": "user", "content": "Hello!"}],
      stream=True
  )
  for chunk in stream:
      if chunk.choices[0].delta.content:
          print(chunk.choices[0].delta.content, end="")

COMMON PARAMETERS:
  model:           Model ID to use
  messages:        Conversation history
  max_tokens:      Maximum response length
  temperature:     Creativity (0.0-2.0)
  top_p:           Nucleus sampling threshold
  stream:          Enable streaming
  stop:            Stop sequences

API KEY:
  If API_KEY is set in .env, include it:
  Authorization: Bearer your-api-key

  If API_KEY is empty, any value works.""",
    },
    "troubleshooting": {
        "title": "Troubleshooting",
        "icon": "\u26a0",
        "content": """Common issues and how to fix them.

SERVER NOT CONNECTED:
  Problem: Red "Disconnected" indicator
  Fixes:
    1. Start the server: python run_platform.py
    2. Check the server URL in Settings
    3. Make sure port 8000 is not blocked
    4. Check firewall settings

MODEL WON'T LOAD:
  Problem: Model stuck on "loading" or shows error
  Fixes:
    1. Check if model files exist at the path
    2. Verify the correct backend is selected
    3. Check available memory (RAM/VRAM)
    4. Try a smaller model first
    5. Check the Activity Log for details

OUT OF MEMORY:
  Problem: CUDA out of memory or system RAM full
  Fixes:
    1. Unload other loaded models first
    2. Use a smaller/quantized model (GGUF)
    3. Reduce batch_size in training
    4. Reduce max_seq_length in training
    5. Use QLoRA instead of LoRA for training

TRAINING FAILS:
  Problem: Training job shows "failed" status
  Fixes:
    1. Check training logs for error details
    2. Verify dataset format is correct
    3. Ensure base model is downloaded
    4. Check dataset has enough samples
    5. Try reducing batch_size

SLOW PERFORMANCE:
  Problem: Model responses are very slow
  Fixes:
    1. Use a smaller model
    2. Use GGUF quantized models (Q4, Q5)
    3. Reduce max_tokens in requests
    4. Close other GPU-using applications
    5. Check if GPU is being used

IMPORT FAILS:
  Problem: Model import shows errors
  Fixes:
    1. Verify ZIP contains valid model files
    2. Check for required files (config.json)
    3. Ensure enough disk space
    4. Try with model_manifest.json included
    5. Check the Import Log for details

PORT ALREADY IN USE:
  Problem: Cannot start server, port busy
  Fixes:
    1. Run: python run_platform.py --kill
    2. Or manually stop the process using the port
    3. Change the port in .env file

GETTING HELP:
  - Check the Activity Log on each page
  - Review server logs in the logs/ folder
  - Visit the API docs at http://localhost:8000/docs
  - Open an issue on GitHub""",
    },
}


class HelpPage(ctk.CTkFrame):
    """Help and documentation page with user-friendly guides."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self._current_section = None

        # Main layout: sidebar + content
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left: Section navigation
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

        # Right: Content area
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)

        # Content header
        self.header_label = ctk.CTkLabel(
            self.content_frame,
            text="",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.header_label.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")

        # Scrollable content text
        self.content_text = ctk.CTkTextbox(
            self.content_frame,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
        )
        self.content_text.grid(row=1, column=0, padx=15, pady=(5, 15), sticky="nsew")

        # Show default section
        self._show_section("quick_start")

    def _show_section(self, section_id: str):
        section = HELP_SECTIONS.get(section_id)
        if not section:
            return

        # Update nav button highlighting
        for sid, btn in self._nav_buttons.items():
            if sid == section_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

        self._current_section = section_id

        # Update header
        self.header_label.configure(text=f"{section['icon']}  {section['title']}")

        # Update content
        self.content_text.delete("1.0", "end")
        self.content_text.insert("1.0", section["content"])
        self.content_text.configure(state="disabled")

    def refresh(self):
        pass
