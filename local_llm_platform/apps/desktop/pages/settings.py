import customtkinter as ctk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


class SettingsPage(ctk.CTkFrame):
    """Settings page - configure server connection and preferences."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app

        ctk.CTkLabel(self, text="Settings", font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(20, 10), sticky="w"
        )

        # Server settings
        server_frame = ctk.CTkFrame(self)
        server_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        server_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(server_frame, text="Server Connection", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w"
        )

        ctk.CTkLabel(server_frame, text="Gateway URL:").grid(row=1, column=0, padx=15, pady=5, sticky="w")
        self.url_entry = ctk.CTkEntry(server_frame, width=300)
        self.url_entry.grid(row=1, column=1, padx=15, pady=5, sticky="ew")
        self.url_entry.insert(0, self.app.api.base_url)

        ctk.CTkButton(
            server_frame, text="Apply", width=80, command=self._apply_url
        ).grid(row=1, column=2, padx=15, pady=5)

        # Theme
        theme_frame = ctk.CTkFrame(self)
        theme_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        theme_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(theme_frame, text="Appearance", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w"
        )

        ctk.CTkLabel(theme_frame, text="Theme:").grid(row=1, column=0, padx=15, pady=5, sticky="w")
        self.theme_combo = ctk.CTkComboBox(
            theme_frame, values=["dark", "light", "system"], width=200,
            command=self._change_theme,
        )
        self.theme_combo.grid(row=1, column=1, padx=15, pady=5, sticky="w")
        self.theme_combo.set("dark")

        ctk.CTkLabel(theme_frame, text="Color:").grid(row=2, column=0, padx=15, pady=5, sticky="w")
        self.color_combo = ctk.CTkComboBox(
            theme_frame, values=["blue", "green", "dark-blue"], width=200,
            command=self._change_color,
        )
        self.color_combo.grid(row=2, column=1, padx=15, pady=5, sticky="w")
        self.color_combo.set("blue")

        # About
        about_frame = ctk.CTkFrame(self)
        about_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(about_frame, text="About", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=15, pady=(15, 5), sticky="w"
        )
        ctk.CTkLabel(
            about_frame,
            text="Local LLM Platform v0.1.0\nDesktop Management UI\nBuilt with CustomTkinter",
            justify="left",
            text_color="gray",
        ).grid(row=1, column=0, padx=15, pady=(0, 15), sticky="w")

        self.grid_columnconfigure(0, weight=1)

    def _apply_url(self):
        new_url = self.url_entry.get()
        if new_url:
            self.app.api.base_url = new_url.rstrip("/")
            self.app.sidebar.set_connected(self.app.api.is_connected())

    def _change_theme(self, value):
        ctk.set_appearance_mode(value)

    def _change_color(self, value):
        ctk.set_default_color_theme(value)
