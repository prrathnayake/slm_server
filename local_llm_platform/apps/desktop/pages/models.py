import customtkinter as ctk
import threading
import time
from datetime import datetime
from tkinter import filedialog
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


class ActivityLog:
    """Shared activity log system for tracking platform events."""

    _instance: Optional["ActivityLog"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._history = []
                    cls._instance._max_history = 1000
                    cls._instance._callbacks = []
        return cls._instance

    def __init__(self):
        pass

    def log(self, level: str, category: str, message: str, details: dict = None):
        """Add an activity log entry."""
        entry = {
            "timestamp": datetime.now(),
            "level": level,
            "category": category,
            "message": message,
            "details": details or {},
        }

        with self._lock:
            self._history.append(entry)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        for callback in self._callbacks:
            try:
                callback(entry)
            except Exception:
                pass

    def info(self, category: str, message: str, details: dict = None):
        self.log("INFO", category, message, details)

    def success(self, category: str, message: str, details: dict = None):
        self.log("SUCCESS", category, message, details)

    def warning(self, category: str, message: str, details: dict = None):
        self.log("WARNING", category, message, details)

    def error(self, category: str, message: str, details: dict = None):
        self.log("ERROR", category, message, details)

    def get_history(self, limit: int = None, category: str = None) -> list:
        """Get activity history, optionally filtered."""
        with self._lock:
            history = self._history.copy()

        if category:
            history = [e for e in history if e["category"] == category]

        if limit:
            history = history[-limit:]

        return history

    def subscribe(self, callback):
        """Subscribe to new log entries."""
        self._callbacks.append(callback)

    def unsubscribe(self, callback):
        """Unsubscribe from log entries."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def clear(self):
        """Clear history."""
        with self._lock:
            self._history.clear()


class ActivityMonitorWindow(ctk.CTkToplevel):
    """Separate window for monitoring activity logs."""

    def __init__(self, parent):
        super().__init__(parent)

        self.title("Activity Monitor")
        self.geometry("900x600")
        self.minsize(700, 400)

        self.activity_log = ActivityLog()
        self.activity_log.subscribe(self._on_new_entry)

        self._setup_ui()
        self._load_history()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text="Activity Monitor", font=ctk.CTkFont(size=18, weight="bold")).pack(
            side="left", padx=10, pady=10
        )

        # Stats
        self.stats_label = ctk.CTkLabel(header, text="")
        self.stats_label.pack(side="right", padx=10, pady=10)

        # Filters
        filter_frame = ctk.CTkFrame(self)
        filter_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        filter_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(filter_frame, text="Filter:").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.filter_combo = ctk.CTkComboBox(
            filter_frame,
            values=["All", "Model", "Inference", "Training", "Import", "System"],
            width=150,
            command=self._apply_filter,
        )
        self.filter_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.filter_combo.set("All")

        ctk.CTkButton(filter_frame, text="Clear", width=60, command=self._clear_history).grid(
            row=0, column=2, padx=5, pady=5
        )

        ctk.CTkButton(filter_frame, text="Export", width=60, command=self._export_logs).grid(
            row=0, column=3, padx=5, pady=5
        )

        # Activity list
        list_frame = ctk.CTkFrame(self)
        list_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)

        # List header
        list_header = ctk.CTkFrame(list_frame, fg_color=("gray80", "gray25"))
        list_header.grid(row=0, column=0, sticky="ew")
        list_header.grid_columnconfigure(0, weight=1)
        list_header.grid_columnconfigure(1, weight=1)
        list_header.grid_columnconfigure(2, weight=3)

        ctk.CTkLabel(list_header, text="Time", font=ctk.CTkFont(size=11, weight="bold")).grid(
            row=0, column=0, padx=8, pady=5, sticky="w"
        )
        ctk.CTkLabel(list_header, text="Category", font=ctk.CTkFont(size=11, weight="weight")).grid(
            row=0, column=1, padx=8, pady=5, sticky="w"
        )
        ctk.CTkLabel(list_header, text="Message", font=ctk.CTkFont(size=11, weight="bold")).grid(
            row=0, column=2, padx=8, pady=5, sticky="w"
        )

        self.activity_list = ctk.CTkScrollableFrame(list_frame)
        self.activity_list.grid(row=1, column=0, sticky="nsew")
        self.activity_list.grid_columnconfigure(0, weight=1)
        self.activity_list.grid_columnconfigure(1, weight=1)
        self.activity_list.grid_columnconfigure(2, weight=3)

    def _load_history(self):
        history = self.activity_log.get_history(limit=200)
        for entry in history:
            self._add_entry_to_list(entry)

        self._update_stats()

    def _apply_filter(self, value):
        for widget in self.activity_list.winfo_children():
            widget.destroy()

        history = self.activity_log.get_history(limit=200)
        filter_val = self.filter_combo.get()

        for entry in history:
            if filter_val == "All" or entry["category"] == filter_val:
                self._add_entry_to_list(entry)

    def _add_entry_to_list(self, entry: dict):
        row = ctk.CTkFrame(self.activity_list, fg_color=("gray90", "gray17"))
        row.pack(fill="x", pady=1)

        level_colors = {
            "INFO": "gray",
            "SUCCESS": "green",
            "WARNING": "orange",
            "ERROR": "red",
        }
        color = level_colors.get(entry["level"], "gray")

        ts = entry["timestamp"].strftime("%H:%M:%S")

        ctk.CTkLabel(row, text=ts, font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=0, column=0, padx=8, pady=5, sticky="w"
        )

        ctk.CTkLabel(
            row, text=f"[{entry['category']}]", font=ctk.CTkFont(size=11, weight="bold"), text_color=color
        ).grid(row=0, column=1, padx=8, pady=5, sticky="w")

        msg_label = ctk.CTkLabel(row, text=entry["message"], anchor="w", font=ctk.CTkFont(size=11))
        msg_label.grid(row=0, column=2, padx=8, pady=5, sticky="ew")

        # Add details tooltip on hover
        if entry["details"]:
            msg_label.bind("<Enter>", lambda e, d=entry["details"]: self._show_details(d))
            msg_label.bind("<Leave>", lambda e: self._hide_details())

    def _show_details(self, details: dict):
        pass

    def _hide_details(self):
        pass

    def _on_new_entry(self, entry: dict):
        self.after(0, lambda: self._add_entry(entry))

    def _add_entry(self, entry: dict):
        filter_val = self.filter_combo.get()
        if filter_val != "All" and entry["category"] != filter_val:
            return

        self._add_entry_to_list(entry)
        self.activity_list._parent_canvas.yview_moveto(1.0)
        self._update_stats()

    def _update_stats(self):
        history = self.activity_log.get_history()
        total = len(history)
        errors = len([e for e in history if e["level"] == "ERROR"])

        self.stats_label.configure(text=f"Total: {total} | Errors: {errors}")

    def _clear_history(self):
        self.activity_log.clear()
        for widget in self.activity_list.winfo_children():
            widget.destroy()
        self._update_stats()

    def _export_logs(self):
        from tkinter import filedialog
        import json

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")],
        )

        if file_path:
            history = self.activity_log.get_history()
            with open(file_path, "w") as f:
                json.dump(history, f, indent=2, default=str)

    def _on_close(self):
        self.activity_log.unsubscribe(self._on_new_entry)
        self.destroy()


class ModelsPage(ctk.CTkFrame):
    """Models management page - list, register, load/unload, adapter merge, download."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self.activity_log = ActivityLog()

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
        ctk.CTkButton(btn_frame, text="Monitor", width=80, command=self._open_monitor).pack(side="left", padx=3)

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

        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew")
        log_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(log_header, text="Activity Log", font=ctk.CTkFont(size=12, weight="bold")).pack(
            side="left", padx=10, pady=(5, 0)
        )

        self.log_counter = ctk.CTkLabel(log_header, text="0 events", text_color="gray", font=ctk.CTkFont(size=10))
        self.log_counter.pack(side="right", padx=10, pady=(5, 0))

        self.log_text = ctk.CTkTextbox(log_frame, height=120, font=ctk.CTkFont(size=11))
        self.log_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.activity_log.subscribe(self._on_activity)

    def _open_monitor(self):
        if not hasattr(self, "_monitor_window") or not self._monitor_window.winfo_exists():
            self._monitor_window = ActivityMonitorWindow(self)
        else:
            self._monitor_window.lift()

    def _on_activity(self, entry: dict):
        self.after(0, lambda: self._add_log_entry(entry))

    def _add_log_entry(self, entry: dict):
        ts = entry["timestamp"].strftime("%H:%M:%S")
        level = entry["level"]
        category = entry["category"]
        message = entry["message"]

        color_map = {"INFO": "gray", "SUCCESS": "green", "WARNING": "orange", "ERROR": "red"}
        color = color_map.get(level, "gray")

        self.log_text.insert("end", f"[{ts}] ", "")
        self.log_text.insert("end", f"[{category}] ", "")
        self.log_text.insert("end", f"{message}\n", "")

        self.log_text.see("end")

        history = self.activity_log.get_history()
        self.log_counter.configure(text=f"{len(history)} events")

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
        self.activity_log.info("Model", "Refreshing model list...")
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            models = self.app.api.list_models()
            count = len(models)
            self.activity_log.success("Model", f"Loaded {count} models")
            self.after(0, lambda: self._populate_table(models))
        except Exception as e:
            self.activity_log.error("Model", f"Error loading models: {e}")

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

        id_text = mid
        if is_adapter:
            id_text = f"[A] {mid}"

        ctk.CTkLabel(row, text=id_text, anchor="w", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, padx=8, pady=5, sticky="w"
        )
        ctk.CTkLabel(row, text=model.get("display_name", ""), anchor="w").grid(
            row=0, column=1, padx=8, pady=5, sticky="w"
        )

        fmt_text = model_format
        if is_adapter and base_model:
            fmt_text = f"{model_format}\n({base_model.split('/')[-1]})"

        ctk.CTkLabel(row, text=fmt_text, anchor="w", font=ctk.CTkFont(size=10)).grid(
            row=0, column=2, padx=8, pady=5, sticky="w"
        )
        ctk.CTkLabel(row, text=backend, anchor="w").grid(row=0, column=3, padx=8, pady=5, sticky="w")
        ctk.CTkLabel(row, text=status, text_color=status_color, anchor="w").grid(
            row=0, column=4, padx=8, pady=5, sticky="w"
        )

        action_frame = ctk.CTkFrame(row, fg_color="transparent")
        action_frame.grid(row=0, column=5, padx=5, pady=2, sticky="e")

        if is_adapter and status == "unloaded":
            ctk.CTkButton(
                action_frame, text="Info", width=50, height=26, fg_color="blue", hover_color="darkblue",
                command=lambda m=mid: self._show_adapter_info(m),
            ).pack(side="left", padx=1)
            ctk.CTkButton(
                action_frame, text="Merge", width=60, height=26, fg_color="purple", hover_color="#4B0082",
                command=lambda m=mid: self._merge_adapter(m),
            ).pack(side="left", padx=1)
        elif status == "unloaded":
            ctk.CTkButton(
                action_frame, text="Load", width=55, height=26,
                command=lambda m=mid, b=backend, p=model.get("artifact_path", ""): self._load_model(m, b, p),
            ).pack(side="left", padx=1)
        elif status == "ready":
            ctk.CTkButton(
                action_frame, text="Unload", width=60, height=26, fg_color="orange", hover_color="darkorange",
                command=lambda m=mid: self._unload_model(m),
            ).pack(side="left", padx=1)
        elif status == "loading":
            ctk.CTkLabel(action_frame, text="Loading...", text_color="orange").pack(side="left", padx=5)

        ctk.CTkButton(
            action_frame, text="\u2715", width=28, height=26, fg_color="red", hover_color="darkred",
            command=lambda m=mid: self._delete_model(m),
        ).pack(side="left", padx=1)

    def _show_adapter_info(self, adapter_id):
        self.activity_log.info("Model", f"Fetching adapter info: {adapter_id}")
        threading.Thread(target=lambda: self._fetch_adapter_info(adapter_id), daemon=True).start()

    def _fetch_adapter_info(self, adapter_id):
        try:
            import httpx
            r = httpx.get(f"{self.app.api.base_url}/v1/adapters/{adapter_id}", timeout=10)
            if r.status_code == 200:
                info = r.json()
                self.activity_log.success("Model", f"Adapter info retrieved: {adapter_id}")
                self.after(0, lambda: self._show_info_dialog(adapter_id, info))
            else:
                self.activity_log.error("Model", f"Failed to get adapter info: {r.status_code}")
        except Exception as e:
            self.activity_log.error("Model", f"Error fetching adapter: {e}")

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
            ctk.CTkLabel(frame, text=value, wraplength=250).grid(row=i, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkButton(dialog, text="Close", command=dialog.destroy, width=100).pack(pady=10)

    def _merge_adapter(self, adapter_id):
        self.activity_log.info("Training", f"Starting merge for adapter: {adapter_id}")
        threading.Thread(target=lambda: self._do_merge(adapter_id), daemon=True).start()

    def _do_merge(self, adapter_id):
        try:
            import httpx
            self.activity_log.info("Training", "Checking base model compatibility...")
            r = httpx.post(f"{self.app.api.base_url}/v1/adapters/{adapter_id}/merge", timeout=300)
            result = r.json()

            if result.get("status") == "merged":
                output_name = result.get("output_name")
                output_path = result.get("output_path")
                self.activity_log.success("Training", f"Merge completed: {output_name}")

                reg_data = {
                    "model_id": output_name,
                    "display_name": f"{adapter_id} (Merged)",
                    "runtime_backend": "transformers",
                    "model_format": "safetensors",
                    "base_model": result.get("base_model"),
                    "artifact_path": output_path,
                    "status": "unloaded",
                }
                self.app.api.register_model(reg_data)
                self.activity_log.success("Model", f"Merged model registered: {output_name}")
                self.refresh()
            elif result.get("status") == "error":
                self.activity_log.error("Training", f"Merge failed: {result.get('message')}")
            else:
                self.activity_log.warning("Training", f"Merge result: {result}")
        except Exception as e:
            self.activity_log.error("Training", f"Merge error: {e}")

    def _load_model(self, model_id, backend, path):
        self.activity_log.info("Model", f"Loading model: {model_id} ({backend})")
        self.activity_log.info("Inference", f"Initializing {backend} runtime...")
        threading.Thread(target=lambda: self._do_load(model_id, backend, path), daemon=True).start()

    def _do_load(self, model_id, backend, path):
        try:
            start_time = time.time()
            result = self.app.api.load_model(model_id, backend, path)
            elapsed = time.time() - start_time
            status = result.get("status", "")

            if status == "adapter_requires_merge":
                self.activity_log.warning("Model", f"Adapter detected - merge required")
                for opt in result.get("options", []):
                    self.activity_log.info("Model", f"  -> {opt.get('description')}")
            elif status == "error":
                self.activity_log.error("Model", f"Load failed: {result.get('error')}")
                if result.get("hint"):
                    self.activity_log.info("Model", f"Hint: {result['hint']}")
            elif status == "loaded":
                self.activity_log.success("Inference", f"Model loaded: {model_id} ({elapsed:.1f}s)")
                self.refresh()
            else:
                self.activity_log.info("Model", f"Result: {result}")
                self.refresh()
        except Exception as e:
            self.activity_log.error("Model", f"Load error: {e}")

    def _unload_model(self, model_id):
        self.activity_log.info("Model", f"Unloading model: {model_id}")
        threading.Thread(target=lambda: self._do_unload(model_id), daemon=True).start()

    def _do_unload(self, model_id):
        try:
            self.app.api.unload_model(model_id)
            self.activity_log.success("Model", f"Model unloaded: {model_id}")
            self.refresh()
        except Exception as e:
            self.activity_log.error("Model", f"Unload error: {e}")

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
            dialog,
            text="Popular models:\n  Qwen/Qwen2.5-1.5B-Instruct\n  meta-llama/Llama-3.2-1B-Instruct\n  microsoft/Phi-3-mini-4k-instruct",
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
        self.activity_log.info("Import", f"Starting download: {model_name}")
        threading.Thread(target=lambda: self._do_download(model_name), daemon=True).start()

    def _do_download(self, model_name):
        try:
            import httpx
            self.activity_log.info("Import", f"Fetching from HuggingFace: {model_name}")
            r = httpx.post(f"{self.app.api.base_url}/v1/models/download", params={"model_name": model_name}, timeout=600)
            result = r.json()
            status = result.get("status", "")

            if status == "downloaded":
                path = result.get("path", "")
                self.activity_log.success("Import", f"Downloaded: {model_name}")
                self.activity_log.info("Import", f"Saved to: {path}")
            elif status == "already_exists":
                self.activity_log.warning("Import", f"Model already exists: {model_name}")
            elif status == "failed":
                self.activity_log.error("Import", f"Download failed: {result.get('error')}")
            else:
                self.activity_log.info("Import", f"Result: {result}")
        except Exception as e:
            self.activity_log.error("Import", f"Download error: {e}")

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
            self.activity_log.info("Model", f"Deleting model: {model_id}")
            try:
                self.app.api.unregister_model(model_id)
                self.activity_log.success("Model", f"Model deleted: {model_id}")
                self.refresh()
            except Exception as e:
                self.activity_log.error("Model", f"Delete error: {e}")

    def _show_register_dialog(self):
        self.activity_log.info("Model", "Opening registration dialog")
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
                self.activity_log.info("Model", f"Registering: {data['model_id']}")
                self.app.api.register_model(data)
                self.activity_log.success("Model", f"Registered: {data['model_id']}")
                dialog.destroy()
                self.refresh()
            except Exception as e:
                self.activity_log.error("Model", f"Registration error: {e}")

        ctk.CTkButton(dialog, text="Register", command=do_register, width=120).pack(pady=15)
