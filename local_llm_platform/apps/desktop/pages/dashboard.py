import customtkinter as ctk
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


class DashboardPage(ctk.CTkFrame):
    """Dashboard showing platform overview and status."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app

        # Title
        title = ctk.CTkLabel(self, text="Dashboard", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 10), sticky="w")

        # Status cards row
        self.cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.cards_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        self.cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.card_models = self._create_card(self.cards_frame, "Models", "0", 0)
        self.card_ready = self._create_card(self.cards_frame, "Ready", "0", 1)
        self.card_training = self._create_card(self.cards_frame, "Training", "0", 2)
        self.card_uptime = self._create_card(self.cards_frame, "Uptime", "--", 3)

        # Server status
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(status_frame, text="Server Status:", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=15, pady=10, sticky="w"
        )
        self.server_status = ctk.CTkLabel(status_frame, text="Checking...", font=ctk.CTkFont(size=14))
        self.server_status.grid(row=0, column=1, padx=15, pady=10, sticky="w")

        self.server_url = ctk.CTkLabel(status_frame, text=self.app.api.base_url, text_color="gray")
        self.server_url.grid(row=1, column=0, columnspan=2, padx=15, pady=(0, 10), sticky="w")

        # Loaded models
        loaded_frame = ctk.CTkFrame(self)
        loaded_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky="nsew")
        loaded_frame.grid_columnconfigure(0, weight=1)
        loaded_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(loaded_frame, text="Loaded Models", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=15, pady=10, sticky="w"
        )

        self.loaded_text = ctk.CTkTextbox(loaded_frame, height=150)
        self.loaded_text.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")

        # Refresh button
        refresh_btn = ctk.CTkButton(self, text="Refresh", command=self.refresh, width=120)
        refresh_btn.grid(row=4, column=0, padx=20, pady=10, sticky="w")

        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _create_card(self, parent, title: str, value: str, col: int) -> ctk.CTkLabel:
        card = ctk.CTkFrame(parent)
        card.grid(row=0, column=col, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(10, 2))
        val_label = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=28, weight="bold"))
        val_label.pack(pady=(0, 10))
        return val_label

    def refresh(self):
        threading.Thread(target=self._refresh_data, daemon=True).start()

    def _refresh_data(self):
        try:
            if not self.app.api.is_connected():
                self.after(0, lambda: self.server_status.configure(text="Disconnected", text_color="red"))
                self.after(0, lambda: self.app.sidebar.set_connected(False))
                return

            self.after(0, lambda: self.server_status.configure(text="Running", text_color="green"))
            self.after(0, lambda: self.app.sidebar.set_connected(True))

            models = self.app.api.list_models()
            ready = len([m for m in models if m.get("status") == "ready"])

            try:
                metrics = self.app.api.get_metrics()
                uptime = metrics.get("uptime_seconds", 0)
                uptime_str = f"{int(uptime // 60)}m"
            except Exception:
                uptime_str = "--"

            try:
                runtime = self.app.api.runtime_health()
                loaded = []
                for backend, info in runtime.items():
                    for mid in info.get("loaded_models", []):
                        loaded.append(f"  {mid} ({backend})")
                loaded_text = "\n".join(loaded) if loaded else "  No models loaded"
            except Exception:
                loaded_text = "  Unable to fetch"

            try:
                jobs = self.app.api.list_training_jobs()
                training_count = len([j for j in jobs if j.get("status") == "running"])
            except Exception:
                training_count = 0

            self.after(0, lambda: self.card_models.configure(text=str(len(models))))
            self.after(0, lambda: self.card_ready.configure(text=str(ready)))
            self.after(0, lambda: self.card_training.configure(text=str(training_count)))
            self.after(0, lambda: self.card_uptime.configure(text=uptime_str))

            self.loaded_text.delete("1.0", "end")
            self.loaded_text.insert("1.0", loaded_text)

        except Exception as e:
            self.after(0, lambda: self.server_status.configure(text=f"Error: {e}", text_color="red"))
            self.after(0, lambda: self.app.sidebar.set_connected(False))
