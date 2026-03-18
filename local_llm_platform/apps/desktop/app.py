from __future__ import annotations

import customtkinter as ctk

from local_llm_platform.apps.desktop.api_client import APIClient
from local_llm_platform.apps.desktop.sidebar import Sidebar
from local_llm_platform.apps.desktop.pages.dashboard import DashboardPage
from local_llm_platform.apps.desktop.pages.models import ModelsPage
from local_llm_platform.apps.desktop.pages.training import TrainingPage
from local_llm_platform.apps.desktop.pages.import_page import ImportPage
from local_llm_platform.apps.desktop.pages.datasets import DatasetsPage
from local_llm_platform.apps.desktop.pages.settings import SettingsPage
from local_llm_platform.apps.desktop.pages.help_page import HelpPage
from local_llm_platform.apps.desktop.pages.hf_browse import HFBrowsePage


class SLMApp(ctk.CTk):
    """Main desktop application window for SLM Platform management."""

    def __init__(self):
        super().__init__()

        self.title("SLM Platform Manager")
        self.geometry("1100x700")
        self.minsize(900, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # API client
        self.api = APIClient()

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = Sidebar(self, on_navigate=self._navigate)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Content area
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        # Pages
        self.pages: dict[str, ctk.CTkFrame] = {}
        self._current_page = None

        self._create_pages()
        self._navigate("dashboard")

        # Auto-refresh connection status
        self._check_connection()

    def _create_pages(self):
        self.pages["dashboard"] = DashboardPage(self.content, self)
        self.pages["models"] = ModelsPage(self.content, self)
        self.pages["training"] = TrainingPage(self.content, self)
        self.pages["import"] = ImportPage(self.content, self)
        self.pages["datasets"] = DatasetsPage(self.content, self)
        self.pages["settings"] = SettingsPage(self.content, self)
        self.pages["help"] = HelpPage(self.content, self)
        self.pages["hf_browse"] = HFBrowsePage(self.content, self)

    def _navigate(self, page_id: str):
        if self._current_page:
            self.pages[self._current_page].grid_forget()

        page = self.pages.get(page_id)
        if page:
            page.grid(row=0, column=0, sticky="nsew")
            self._current_page = page_id

            # Auto-refresh on navigate
            if hasattr(page, "refresh"):
                page.refresh()

    def _check_connection(self):
        try:
            connected = self.api.is_connected()
            self.sidebar.set_connected(connected)
        except Exception:
            self.sidebar.set_connected(False)
        self.after(8000, self._check_connection)


def main():
    app = SLMApp()
    app.mainloop()


if __name__ == "__main__":
    main()
