import customtkinter as ctk
import threading
from tkinter import filedialog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


class ImportPage(ctk.CTkFrame):
    """Import page - import models from zip files."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app

        # Title
        ctk.CTkLabel(self, text="Import Models", font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(20, 10), sticky="w"
        )

        # Import card
        card = ctk.CTkFrame(self)
        card.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card, text="Import from ZIP", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w"
        )

        ctk.CTkLabel(card, text="ZIP File:").grid(row=1, column=0, padx=15, pady=5, sticky="w")
        self.file_path = ctk.CTkEntry(card, width=300)
        self.file_path.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(card, text="Browse", width=80, command=self._browse_file).grid(
            row=1, column=2, padx=15, pady=5
        )

        ctk.CTkLabel(card, text="Model ID (optional):").grid(row=2, column=0, padx=15, pady=5, sticky="w")
        self.model_id_entry = ctk.CTkEntry(card, width=300)
        self.model_id_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(
            card,
            text="Import Model",
            command=self._import_model,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=3, column=0, columnspan=3, padx=15, pady=15, sticky="ew")

        # Info
        info_frame = ctk.CTkFrame(self)
        info_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(info_frame, text="Supported Formats", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=15, pady=(10, 5), sticky="w"
        )

        info_text = (
            "Full HuggingFace model folder (safetensors)\n"
            "LoRA adapter folder\n"
            "Merged safetensors model\n"
            "GGUF model folder\n"
            "Tokenizer bundle\n"
            "Include model_manifest.json for best results"
        )
        ctk.CTkLabel(info_frame, text=info_text, justify="left", text_color="gray").grid(
            row=1, column=0, padx=15, pady=(0, 10), sticky="w"
        )

        # Log
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(log_frame, text="Import Log", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=0, padx=10, pady=(5, 0), sticky="w"
        )
        self.log_text = ctk.CTkTextbox(log_frame, height=200)
        self.log_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select model ZIP",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
        )
        if path:
            self.file_path.delete(0, "end")
            self.file_path.insert(0, path)

    def _import_model(self):
        zip_path = self.file_path.get()
        if not zip_path:
            self._log("Please select a ZIP file")
            return

        model_id = self.model_id_entry.get() or None
        threading.Thread(
            target=lambda: self._do_import(zip_path, model_id), daemon=True
        ).start()

    def _do_import(self, zip_path, model_id):
        try:
            self._log(f"Importing {zip_path}...")
            result = self.app.api.import_model_zip(zip_path, model_id)

            if result.get("success"):
                self._log(f"Success! Model ID: {result.get('model_id')}")
                self._log(f"Format: {result.get('detected_format')}")
                self._log(f"Path: {result.get('artifact_path')}")
                if result.get("warnings"):
                    for w in result["warnings"]:
                        self._log(f"Warning: {w}")
            else:
                for e in result.get("errors", []):
                    self._log(f"Error: {e}")

        except Exception as e:
            self._log(f"Import failed: {e}")

    def _log(self, msg: str):
        def _do():
            self.log_text.insert("end", f"{msg}\n")
            self.log_text.see("end")

        self.after(0, _do)
