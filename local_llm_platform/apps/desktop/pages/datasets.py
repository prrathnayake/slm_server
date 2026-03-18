import customtkinter as ctk
import threading
from tkinter import filedialog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


class DatasetsPage(ctk.CTkFrame):
    """Datasets management page."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="Datasets", font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, sticky="w"
        )

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(btn_frame, text="Refresh", width=100, command=self.refresh).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="+ Upload", width=100, command=self._upload_dialog).pack(side="left", padx=5)

        # Datasets list
        self.list_frame = ctk.CTkScrollableFrame(self)
        self.list_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.list_frame.grid_columnconfigure(0, weight=1)

        # Upload section
        upload_frame = ctk.CTkFrame(self)
        upload_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        upload_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(upload_frame, text="Quick Upload", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=15, pady=(10, 5), sticky="w"
        )

        ctk.CTkLabel(upload_frame, text="File:").grid(row=1, column=0, padx=15, pady=5, sticky="w")
        self.file_path = ctk.CTkEntry(upload_frame, width=300)
        self.file_path.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(upload_frame, text="...", width=30, command=self._browse_file).grid(
            row=1, column=2, padx=15, pady=5
        )

        ctk.CTkLabel(upload_frame, text="Name:").grid(row=2, column=0, padx=15, pady=5, sticky="w")
        self.name_entry = ctk.CTkEntry(upload_frame, width=300)
        self.name_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(upload_frame, text="Format:").grid(row=3, column=0, padx=15, pady=5, sticky="w")
        self.format_combo = ctk.CTkComboBox(
            upload_frame, values=["jsonl", "chat", "instruction", "tool_call", "plain_text"], width=300
        )
        self.format_combo.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.format_combo.set("jsonl")

        ctk.CTkButton(upload_frame, text="Upload", command=self._do_upload, width=120).grid(
            row=4, column=0, columnspan=3, padx=15, pady=10
        )

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.refresh()

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select dataset file",
            filetypes=[
                ("JSONL", "*.jsonl"),
                ("JSON", "*.json"),
                ("Text", "*.txt"),
                ("CSV", "*.csv"),
                ("All", "*.*"),
            ],
        )
        if path:
            self.file_path.delete(0, "end")
            self.file_path.insert(0, path)

    def refresh(self):
        threading.Thread(target=self._load_datasets, daemon=True).start()

    def _load_datasets(self):
        try:
            datasets = self.app.api.list_datasets()
            self.after(0, lambda: self._populate(datasets))
        except Exception as e:
            pass

    def _populate(self, datasets):
        for w in self.list_frame.winfo_children():
            w.destroy()

        if not datasets:
            ctk.CTkLabel(self.list_frame, text="No datasets uploaded", text_color="gray").pack(pady=30)
            return

        # Header
        header = ctk.CTkFrame(self.list_frame, fg_color=("gray80", "gray25"))
        header.pack(fill="x", pady=(0, 5))
        for i, text in enumerate(["ID", "Name", "Format", "Samples"]):
            ctk.CTkLabel(header, text=text, font=ctk.CTkFont(size=12, weight="bold")).grid(
                row=0, column=i, padx=10, pady=5, sticky="w"
            )
            header.grid_columnconfigure(i, weight=1)

        for ds in datasets:
            row = ctk.CTkFrame(self.list_frame, fg_color=("gray90", "gray17"))
            row.pack(fill="x", pady=1)
            for i, key in enumerate(["dataset_id", "name", "format", "num_samples"]):
                ctk.CTkLabel(row, text=str(ds.get(key, "")), anchor="w").grid(
                    row=0, column=i, padx=10, pady=5, sticky="w"
                )
                row.grid_columnconfigure(i, weight=1)

    def _upload_dialog(self):
        self._browse_file()

    def _do_upload(self):
        path = self.file_path.get()
        name = self.name_entry.get()
        fmt = self.format_combo.get()

        if not path or not name:
            return

        threading.Thread(
            target=lambda: self._perform_upload(path, name, fmt), daemon=True
        ).start()

    def _perform_upload(self, path, name, fmt):
        try:
            result = self.app.api.upload_dataset(path, name, fmt)
            self.after(0, self.refresh)
        except Exception:
            pass
