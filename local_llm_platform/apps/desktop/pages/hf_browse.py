import customtkinter as ctk
import threading
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


class HFBrowsePage(ctk.CTkFrame):
    """HuggingFace Model Browser - search, browse, and download free models."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self._current_models: List[Dict[str, Any]] = []
        self._download_buttons: Dict[str, ctk.CTkButton] = {}

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="HuggingFace Model Browser", font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, sticky="w"
        )

        # Search bar
        search_frame = ctk.CTkFrame(self)
        search_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        search_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(search_frame, text="Search:").grid(row=0, column=0, padx=(10, 5), pady=8)
        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="e.g. Qwen, Llama, Mistral...")
        self.search_entry.grid(row=0, column=1, padx=5, pady=8, sticky="ew")
        self.search_entry.bind("<Return>", lambda e: self._do_search())

        ctk.CTkButton(search_frame, text="Search", width=80, command=self._do_search).grid(
            row=0, column=2, padx=5, pady=8
        )
        ctk.CTkButton(search_frame, text="Popular", width=80, fg_color="green", hover_color="darkgreen",
                       command=self._load_popular).grid(row=0, column=3, padx=5, pady=8)

        # Filter options
        filter_frame = ctk.CTkFrame(self, fg_color="transparent")
        filter_frame.grid(row=2, column=0, padx=20, pady=(0, 5), sticky="ew")

        ctk.CTkLabel(filter_frame, text="Sort:").pack(side="left", padx=(0, 5))
        self.sort_combo = ctk.CTkComboBox(filter_frame, values=["downloads", "likes", "lastModified"], width=130)
        self.sort_combo.pack(side="left", padx=5)
        self.sort_combo.set("downloads")

        ctk.CTkLabel(filter_frame, text="Size:").pack(side="left", padx=(15, 5))
        self.size_filter = ctk.CTkComboBox(filter_frame, values=["All", "Small (<1B)", "Medium (1-3B)", "Large (3-7B)", "XL (>7B)"], width=150)
        self.size_filter.pack(side="left", padx=5)
        self.size_filter.set("All")

        # Models list
        self.table_frame = ctk.CTkScrollableFrame(self)
        self.table_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        self.table_frame.grid_columnconfigure(0, weight=1)

        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Log area
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        log_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(log_frame, text="Download Log", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=0, padx=10, pady=(5, 0), sticky="w"
        )
        self.log_text = ctk.CTkTextbox(log_frame, height=80)
        self.log_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        # Load popular models by default
        self._load_popular()

    def refresh(self):
        self._load_popular()

    def _load_popular(self):
        threading.Thread(target=self._fetch_popular, daemon=True).start()

    def _fetch_popular(self):
        try:
            import httpx
            r = httpx.get(f"{self.app.api.base_url}/v1/huggingface/popular", timeout=15)
            if r.status_code == 200:
                models = r.json().get("models", [])
                self.after(0, lambda: self._populate_table(models, is_popular=True))
        except Exception as e:
            self._log(f"Error loading popular models: {e}")

    def _do_search(self):
        query = self.search_entry.get().strip()
        sort = self.sort_combo.get()
        threading.Thread(target=lambda: self._fetch_search(query, sort), daemon=True).start()

    def _fetch_search(self, query, sort):
        try:
            import httpx
            params = {"query": query, "sort": sort, "limit": 30}
            r = httpx.get(f"{self.app.api.base_url}/v1/huggingface/search", params=params, timeout=30)
            if r.status_code == 200:
                models = r.json().get("models", [])
                self.after(0, lambda: self._populate_table(models, is_popular=False))
            else:
                self._log(f"Search failed: {r.text}")
        except Exception as e:
            self._log(f"Error: {e}")

    def _populate_table(self, models, is_popular=False):
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self._download_buttons.clear()

        if not models:
            ctk.CTkLabel(self.table_frame, text="No models found").grid(row=0, column=0, pady=20)
            return

        # Category label
        label_text = "Popular Models for Fine-Tuning" if is_popular else f"Search Results ({len(models)})"
        ctk.CTkLabel(
            self.table_frame, text=label_text,
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, padx=10, pady=(5, 10), sticky="w")

        # Header
        header = ctk.CTkFrame(self.table_frame, fg_color=("gray80", "gray25"))
        header.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        header.grid_columnconfigure(0, weight=3)
        header.grid_columnconfigure(1, weight=1)
        header.grid_columnconfigure(2, weight=1)
        header.grid_columnconfigure(3, weight=2)

        for i, text in enumerate(["Model", "Downloads", "Likes", "Tags", "Action"]):
            ctk.CTkLabel(header, text=text, font=ctk.CTkFont(size=11, weight="bold")).grid(
                row=0, column=i, padx=6, pady=4, sticky="w"
            )

        self._current_models = models
        size_filter = self.size_filter.get()

        row_idx = 2
        for model in models:
            if not self._matches_size_filter(model, size_filter):
                continue
            self._create_model_row(model, row_idx)
            row_idx += 1

    def _matches_size_filter(self, model, size_filter):
        if size_filter == "All":
            return True
        params = model.get("params", "")
        if not params:
            return True
        params_lower = params.lower()
        try:
            if params_lower.endswith("m"):
                num = float(params_lower[:-1])
                actual_b = num / 1000
            else:
                actual_b = float(params_lower.replace("b", ""))
        except (ValueError, AttributeError):
            return True

        if size_filter == "Small (<1B)":
            return actual_b < 1
        elif size_filter == "Medium (1-3B)":
            return 1 <= actual_b < 3
        elif size_filter == "Large (3-7B)":
            return 3 <= actual_b <= 7
        elif size_filter == "XL (>7B)":
            return actual_b > 7
        return True

    def _create_model_row(self, model, row_idx):
        row = ctk.CTkFrame(self.table_frame, fg_color=("gray92", "gray17"))
        row.grid(row=row_idx, column=0, sticky="ew", pady=1)
        row.grid_columnconfigure(0, weight=3)
        row.grid_columnconfigure(1, weight=1)
        row.grid_columnconfigure(2, weight=1)
        row.grid_columnconfigure(3, weight=2)

        model_id = model.get("id") or model.get("name", "")
        author = model.get("author", "")
        params = model.get("params", "")
        downloads = model.get("downloads", 0)
        likes = model.get("likes", 0)
        tags = model.get("tags", [])
        task = model.get("task", "")

        # Model name with params
        name_text = model_id
        if params:
            name_text = f"{model_id}  ({params})"

        ctk.CTkLabel(row, text=name_text, anchor="w", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, padx=8, pady=6, sticky="w"
        )

        # Downloads
        dl_text = self._format_number(downloads)
        ctk.CTkLabel(row, text=dl_text, anchor="w").grid(
            row=0, column=1, padx=8, pady=6, sticky="w"
        )

        # Likes
        lk_text = self._format_number(likes)
        ctk.CTkLabel(row, text=lk_text, anchor="w").grid(
            row=0, column=2, padx=8, pady=6, sticky="w"
        )

        # Tags (show first 2-3 relevant tags)
        relevant_tags = [t for t in tags if t in (
            "pytorch", "safetensors", "transformers", "gguf", "lora",
            "text-generation", "conversational", "chat", "instruct"
        )][:3]
        tags_text = ", ".join(relevant_tags) if relevant_tags else task
        ctk.CTkLabel(row, text=tags_text, anchor="w", font=ctk.CTkFont(size=10), text_color="gray").grid(
            row=0, column=3, padx=8, pady=6, sticky="w"
        )

        # Download button
        btn = ctk.CTkButton(
            row, text="Download", width=90, height=28,
            fg_color="green", hover_color="darkgreen",
            command=lambda m=model_id: self._download_model(m),
        )
        btn.grid(row=0, column=4, padx=8, pady=4)
        self._download_buttons[model_id] = btn

    def _format_number(self, num):
        if num >= 1000000:
            return f"{num/1000000:.1f}M"
        elif num >= 1000:
            return f"{num/1000:.0f}K"
        return str(num)

    def _download_model(self, model_id):
        btn = self._download_buttons.get(model_id)
        if btn:
            btn.configure(state="disabled", text="Downloading...", fg_color="orange")
        self._log(f"Downloading {model_id}...")
        threading.Thread(target=lambda: self._do_download(model_id), daemon=True).start()

    def _do_download(self, model_id):
        try:
            import httpx
            r = httpx.post(
                f"{self.app.api.base_url}/v1/huggingface/models/{model_id}/download",
                timeout=1200,
            )
            result = r.json()
            status = result.get("status", "")

            btn = self._download_buttons.get(model_id)

            if status == "downloaded":
                self._log(f"Downloaded: {model_id} -> {result.get('path', '')}")
                self.after(0, lambda: btn.configure(text="Downloaded", fg_color="gray50") if btn else None)
            elif status == "already_exists":
                self._log(f"Already exists: {model_id}")
                self.after(0, lambda: btn.configure(text="Exists", fg_color="blue") if btn else None)
            elif status == "failed":
                self._log(f"Failed: {result.get('error', 'Unknown error')}")
                self.after(0, lambda: btn.configure(text="Retry", fg_color="green", state="normal") if btn else None)
            else:
                self._log(f"Result: {result}")
                self.after(0, lambda: btn.configure(text="Download", fg_color="green", state="normal") if btn else None)
        except Exception as e:
            self._log(f"Error: {e}")
            btn = self._download_buttons.get(model_id)
            self.after(0, lambda: btn.configure(text="Retry", fg_color="green", state="normal") if btn else None)

    def _log(self, msg: str):
        def _do():
            self.log_text.insert("end", f"{msg}\n")
            self.log_text.see("end")
        self.after(0, _do)
