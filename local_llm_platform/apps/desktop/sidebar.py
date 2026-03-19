import customtkinter as ctk
from typing import Callable, Optional


class Sidebar(ctk.CTkFrame):
    """Navigation sidebar with page buttons."""

    def __init__(self, master, on_navigate: Callable[[str], None], **kwargs):
        super().__init__(master, width=200, corner_radius=0, **kwargs)
        self.on_navigate = on_navigate
        self._buttons: dict[str, ctk.CTkButton] = {}
        self._active_page = "dashboard"

        # Navigation buttons
        pages = [
            ("dashboard", "Dashboard", "\u2302"),
            ("monitoring", "Monitor", "\u25A0"),
            ("models", "Models", "\u2B21"),
            ("hf_browse", "HF Browser", "\U0001F917"),
            ("training", "Training", "\u2699"),
            ("import", "Import", "\u2913"),
            ("datasets", "Datasets", "\u2637"),
            ("settings", "Settings", "\u2691"),
            ("help", "Help & Docs", "\u2753"),
        ]

        # Title
        title = ctk.CTkLabel(
            self,
            text="SLM Platform",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        subtitle = ctk.CTkLabel(
            self,
            text="v0.1.0",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

        for i, (page_id, label, icon) in enumerate(pages):
            btn = ctk.CTkButton(
                self,
                text=f"  {icon}  {label}",
                anchor="w",
                height=40,
                font=ctk.CTkFont(size=14),
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                command=lambda p=page_id: self._navigate(p),
            )
            btn.grid(row=i + 2, column=0, padx=10, pady=2, sticky="ew")
            self._buttons[page_id] = btn

        # Status indicator
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.grid(row=98, column=0, padx=10, pady=10, sticky="sew")

        self.status_dot = ctk.CTkLabel(
            self.status_frame,
            text="\u25CF",
            text_color="red",
            font=ctk.CTkFont(size=12),
        )
        self.status_dot.pack(side="left", padx=(5, 5))

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Disconnected",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.status_label.pack(side="left")

        self._set_active("dashboard")

    def _navigate(self, page_id: str):
        self._set_active(page_id)
        self.on_navigate(page_id)

    def _set_active(self, page_id: str):
        for pid, btn in self._buttons.items():
            if pid == page_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")
        self._active_page = page_id

    def set_connected(self, connected: bool):
        if connected:
            self.status_dot.configure(text_color="green")
            self.status_label.configure(text="Connected")
        else:
            self.status_dot.configure(text_color="red")
            self.status_label.configure(text="Disconnected")
