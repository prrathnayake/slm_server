import customtkinter as ctk
import threading
import time
from tkinter import filedialog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


class ModelsPage(ctk.CTkFrame):
    """Models management page - list, register, load/unload, adapter merge, download."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app

        # Title and buttons
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="Models", font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, sticky="w"
        )

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        ctk.CTkButton(btn_frame, text="Refresh", width=90, command=self.refresh).pack(side="left", padx=3)
        ctk.CTkButton(btn_frame, text="Download Base", width=110, command=self._download_dialog).pack(side="left", padx=3)
        ctk.CTkButton(btn_frame, text="+ Register", width=90, command=self._show_register_dialog).pack(side="left", padx=3)

        # Models table
        self.table_frame = ctk.CTkScrollableFrame(self)
        self.table_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        self.table_frame.grid_columnconfigure(0, weight=1)

        self._create_table_header()

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Log area
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew")
        log_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(log_frame, text="Activity Log", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=0, padx=10, pady=(5, 0), sticky="w"
        )
        self.log_text = ctk.CTkTextbox(log_frame, height=100)
        self.log_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

    def _create_table_header(self):
        header = ctk.CTkFrame(self.table_frame, fg_color=("gray80", "gray25"))
        header.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        header.grid_columnconfigure(1, weight=2)
        header.grid_columnconfigure(2, weight=1)
        header.grid_columnconfigure(3, weight=1)
        header.grid_columnconfigure(4, weight=1)

        for i, text in enumerate(["Model ID", "Display Name", "Format", "Backend", "Status", "Actions"]):
            ctk.CTkLabel(header, text=text, font=ctk.CTkFont(size=12, weight="bold")).grid(
                row=0, column=i, padx=8, pady=5, sticky="w"
            )

    def refresh(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            models = self.app.api.list_models()
            self.after(0, lambda: self._populate_table(models))
        except Exception as e:
            self._log(f"Error loading models: {e}")

    def _populate_table(self, models):
        for widget in self.table_frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                widget.destroy()

        self._create_table_header()

        for i, model in enumerate(models):
            self._create_model_row(model, i + 1)

    def _create_model_row(self, model, row_idx):
        row = ctk.CTkFrame(self.table_frame, fg_color=("gray90", "gray17"))
        row.grid(row=row_idx, column=0, sticky="ew", pady=1)
        row.grid_columnconfigure(1, weight=2)
        row.grid_columnconfigure(2, weight=1)
        row.grid_columnconfigure(3, weight=1)
        row.grid_columnconfigure(4, weight=1)

        mid = model.get("model_id", "")
        status = model.get("status", "unknown")
        model_format = model.get("model_format", "")
        backend = model.get("runtime_backend", "")
        base_model = model.get("base_model", "")
        is_adapter = model_format == "adapter"

        status_color = {
            "ready": "green",
            "loading": "orange",
            "training": "blue",
            "failed": "red",
            "unloaded": "gray",
        }.get(status, "gray")

        # Model ID with adapter indicator
        id_text = mid
        if is_adapter:
            id_text = f"[A] {mid}"

        ctk.CTkLabel(row, text=id_text, anchor="w", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, padx=8, pady=5, sticky="w"
        )
        ctk.CTkLabel(row, text=model.get("display_name", ""), anchor="w").grid(
            row=0, column=1, padx=8, pady=5, sticky="w"
        )

        # Format with base model info for adapters
        fmt_text = model_format
        if is_adapter and base_model:
            fmt_text = f"{model_format}\n({base_model.split('/')[-1]})"

        ctk.CTkLabel(row, text=fmt_text, anchor="w", font=ctk.CTkFont(size=10)).grid(
            row=0, column=2, padx=8, pady=5, sticky="w"
        )
        ctk.CTkLabel(row, text=backend, anchor="w").grid(
            row=0, column=3, padx=8, pady=5, sticky="w"
        )
        ctk.CTkLabel(row, text=status, text_color=status_color, anchor="w").grid(
            row=0, column=4, padx=8, pady=5, sticky="w"
        )

        # Action buttons
        action_frame = ctk.CTkFrame(row, fg_color="transparent")
        action_frame.grid(row=0, column=5, padx=5, pady=2, sticky="e")

        if is_adapter and status == "unloaded":
            # Adapter-specific buttons
            ctk.CTkButton(
                action_frame, text="Info", width=50, height=26,
                fg_color="blue", hover_color="darkblue",
                command=lambda m=mid: self._show_adapter_info(m),
            ).pack(side="left", padx=1)
            ctk.CTkButton(
                action_frame, text="Merge", width=60, height=26,
                fg_color="purple", hover_color="#4B0082",
                command=lambda m=mid: self._merge_adapter(m),
            ).pack(side="left", padx=1)
        elif status == "unloaded":
            ctk.CTkButton(
                action_frame, text="Load", width=55, height=26,
                command=lambda m=mid, b=backend, p=model.get("artifact_path", ""): self._load_model(m, b, p),
            ).pack(side="left", padx=1)
        elif status == "ready":
            ctk.CTkButton(
                action_frame, text="Unload", width=60, height=26,
                fg_color="orange", hover_color="darkorange",
                command=lambda m=mid: self._unload_model(m),
            ).pack(side="left", padx=1)
        elif status == "loading":
            ctk.CTkLabel(action_frame, text="Loading...", text_color="orange").pack(side="left", padx=5)

        ctk.CTkButton(
            action_frame, text="\u2715", width=28, height=26,
            fg_color="red", hover_color="darkred",
            command=lambda m=mid: self._delete_model(m),
        ).pack(side="left", padx=1)

    # --- Adapter Info Dialog ---

    def _show_adapter_info(self, adapter_id):
        threading.Thread(
            target=lambda: self._fetch_adapter_info(adapter_id), daemon=True
        ).start()

    def _fetch_adapter_info(self, adapter_id):
        try:
            import httpx
            r = httpx.get(f"{self.app.api.base_url}/v1/adapters/{adapter_id}", timeout=10)
            if r.status_code == 200:
                info = r.json()
                self.after(0, lambda: self._show_info_dialog(adapter_id, info))
            else:
                self._log(f"Failed to get adapter info: {r.text}")
        except Exception as e:
            self._log(f"Error: {e}")

    def _show_info_dialog(self, adapter_id, info):
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Adapter: {adapter_id}")
        dialog.geometry("400x350")
        dialog.resizable(False, False)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text=f"Adapter: {adapter_id}", font=ctk.CTkFont(size=16, weight="bold")).pack(
            pady=(15, 10)
        )

        frame = ctk.CTkFrame(dialog)
        frame.pack(padx=20, pady=10, fill="both", expand=True)

        rows = [
            ("Base Model", info.get("base_model", "N/A")),
            ("Type", info.get("peft_type", "N/A")),
            ("Rank (r)", str(info.get("r", "N/A"))),
            ("Alpha", str(info.get("lora_alpha", "N/A"))),
            ("Targets", ", ".join(info.get("target_modules", []))),
            ("Path", info.get("adapter_path", "N/A")),
        ]

        for i, (label, value) in enumerate(rows):
            ctk.CTkLabel(frame, text=f"{label}:", font=ctk.CTkFont(weight="bold")).grid(
                row=i, column=0, padx=10, pady=5, sticky="w"
            )
            ctk.CTkLabel(frame, text=value, wraplength=250).grid(
                row=i, column=1, padx=10, pady=5, sticky="w"
            )

        ctk.CTkButton(
            dialog, text="Close", command=dialog.destroy, width=100
        ).pack(pady=10)

    # --- Merge Adapter ---

    def _merge_adapter(self, adapter_id):
        self._log(f"Merging adapter {adapter_id}...")
        threading.Thread(
            target=lambda: self._do_merge(adapter_id), daemon=True
        ).start()

    def _do_merge(self, adapter_id):
        try:
            import httpx
            self._log("Step 1/2: Checking base model...")
            r = httpx.post(
                f"{self.app.api.base_url}/v1/adapters/{adapter_id}/merge",
                timeout=300,
            )
            result = r.json()

            if result.get("status") == "merged":
                self._log(f"Merged! Output: {result.get('output_name')}")
                self._log(f"Path: {result.get('output_path')}")
                self._log("Registering merged model...")

                # Register merged model
                reg_data = {
                    "model_id": result["output_name"],
                    "display_name": f"{adapter_id} (Merged)",
                    "runtime_backend": "transformers",
                    "model_format": "safetensors",
                    "base_model": result.get("base_model"),
                    "artifact_path": result.get("output_path"),
                    "status": "unloaded",
                }
                self.app.api.register_model(reg_data)
                self._log("Merged model registered!")
                self.refresh()
            elif result.get("status") == "error":
                self._log(f"Merge failed: {result.get('message')}")
            else:
                self._log(f"Merge result: {result}")
        except Exception as e:
            self._log(f"Error: {e}")

    # --- Load Model ---

    def _load_model(self, model_id, backend, path):
        self._log(f"Loading {model_id} ({backend})...")
        threading.Thread(
            target=lambda: self._do_load(model_id, backend, path), daemon=True
        ).start()

    def _do_load(self, model_id, backend, path):
        try:
            result = self.app.api.load_model(model_id, backend, path)
            status = result.get("status", "")

            if status == "adapter_requires_merge":
                self._log(f"Adapter detected - needs merge first")
                for opt in result.get("options", []):
                    self._log(f"  -> {opt.get('description')}")
            elif status == "error":
                self._log(f"Error: {result.get('error')}")
                if result.get("hint"):
                    self._log(f"Hint: {result['hint']}")
            elif status == "loaded":
                self._log(f"Loaded {model_id}!")
                self.refresh()
            else:
                self._log(f"Result: {result}")
                self.refresh()
        except Exception as e:
            self._log(f"Error: {e}")

    def _unload_model(self, model_id):
        self._log(f"Unloading {model_id}...")
        threading.Thread(
            target=lambda: self._do_unload(model_id), daemon=True
        ).start()

    def _do_unload(self, model_id):
        try:
            self.app.api.unload_model(model_id)
            self._log(f"Unloaded {model_id}")
            self.refresh()
        except Exception as e:
            self._log(f"Error: {e}")

    # --- Download Base Model ---

    def _download_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Download Base Model")
        dialog.geometry("450x250")
        dialog.resizable(False, False)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Download Base Model", font=ctk.CTkFont(size=18, weight="bold")).pack(
            pady=(15, 10)
        )

        ctk.CTkLabel(dialog, text="Model name from HuggingFace:").pack(pady=(10, 5))

        entry = ctk.CTkEntry(dialog, width=350, placeholder_text="e.g. Qwen/Qwen2.5-1.5B-Instruct")
        entry.pack(pady=5)

        ctk.CTkLabel(
            dialog, text="Popular models:\n  Qwen/Qwen2.5-1.5B-Instruct\n  meta-llama/Llama-3.2-1B-Instruct\n  microsoft/Phi-3-mini-4k-instruct",
            justify="left", text_color="gray", font=ctk.CTkFont(size=11)
        ).pack(pady=5)

        def do_download():
            model_name = entry.get().strip()
            if not model_name:
                return
            dialog.destroy()
            self._download_model(model_name)

        ctk.CTkButton(dialog, text="Download", command=do_download, width=120).pack(pady=10)

    def _download_model(self, model_name):
        self._log(f"Downloading {model_name}...")
        threading.Thread(
            target=lambda: self._do_download(model_name), daemon=True
        ).start()

    def _do_download(self, model_name):
        try:
            import httpx
            r = httpx.post(
                f"{self.app.api.base_url}/v1/models/download",
                params={"model_name": model_name},
                timeout=600,
            )
            result = r.json()
            status = result.get("status", "")

            if status == "downloaded":
                self._log(f"Downloaded: {result.get('path')}")
            elif status == "already_exists":
                self._log(f"Already exists: {result.get('path')}")
            elif status == "failed":
                self._log(f"Download failed: {result.get('error')}")
            else:
                self._log(f"Result: {result}")
        except Exception as e:
            self._log(f"Error: {e}")

    # --- Delete ---

    def _delete_model(self, model_id):
        from CTkMessagebox import CTkMessagebox
        msg = CTkMessagebox(
            title="Delete Model",
            message=f"Delete model '{model_id}' from registry?",
            icon="warning",
            option_1="Cancel",
            option_2="Delete",
        )
        if msg.get() == "Delete":
            try:
                self.app.api.unregister_model(model_id)
                self._log(f"Deleted {model_id}")
                self.refresh()
            except Exception as e:
                self._log(f"Error: {e}")

    # --- Register Dialog ---

    def _show_register_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Register Model")
        dialog.geometry("450x400")
        dialog.resizable(False, False)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Register New Model", font=ctk.CTkFont(size=18, weight="bold")).pack(
            pady=(15, 10)
        )

        form = ctk.CTkFrame(dialog, fg_color="transparent")
        form.pack(padx=20, pady=10, fill="x")

        fields = {}

        labels = [
            ("model_id", "Model ID:", "entry"),
            ("display_name", "Display Name:", "entry"),
            ("backend", "Backend:", "combo", ["llama_cpp", "vllm", "transformers", "tgi", "remote_http", "remote_ssh"]),
            ("format", "Format:", "combo", ["gguf", "safetensors", "adapter"]),
            ("specialization", "Specialization:", "combo", ["general", "reasoning", "planning", "tool-calling", "personality", "coder"]),
        ]

        for i, label_info in enumerate(labels):
            field_id = label_info[0]
            label_text = label_info[1]
            field_type = label_info[2]

            ctk.CTkLabel(form, text=label_text).grid(row=i, column=0, padx=5, pady=5, sticky="w")

            if field_type == "entry":
                widget = ctk.CTkEntry(form, width=250)
                widget.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            elif field_type == "combo":
                widget = ctk.CTkComboBox(form, values=label_info[3], width=250)
                widget.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
                widget.set(label_info[3][0])

            fields[field_id] = widget

        # Artifact path
        row = len(labels)
        ctk.CTkLabel(form, text="Artifact Path:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        path_frame = ctk.CTkFrame(form, fg_color="transparent")
        path_frame.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        fields["artifact_path"] = ctk.CTkEntry(path_frame, width=180)
        fields["artifact_path"].pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            path_frame, text="...", width=30,
            command=lambda: fields["artifact_path"].insert(0, filedialog.askdirectory() or ""),
        ).pack(side="right", padx=(5, 0))

        form.grid_columnconfigure(1, weight=1)

        def do_register():
            data = {
                "model_id": fields["model_id"].get(),
                "display_name": fields["display_name"].get() or fields["model_id"].get(),
                "runtime_backend": fields["backend"].get(),
                "model_format": fields["format"].get(),
                "specialization": fields["specialization"].get(),
                "artifact_path": fields["artifact_path"].get() or None,
                "status": "unloaded",
            }
            if not data["model_id"]:
                return
            try:
                self.app.api.register_model(data)
                self._log(f"Registered {data['model_id']}")
                dialog.destroy()
                self.refresh()
            except Exception as e:
                self._log(f"Error: {e}")

        ctk.CTkButton(dialog, text="Register", command=do_register, width=120).pack(pady=15)

    def _log(self, msg: str):
        def _do():
            self.log_text.insert("end", f"{msg}\n")
            self.log_text.see("end")

        self.after(0, _do)
