import customtkinter as ctk
import threading
import time
import psutil
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


class MonitoringPage(ctk.CTkFrame):
    """Comprehensive monitoring page for platform metrics and system resources."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self._polling = False
        self._polling_interval = 2000  # 2 seconds

        self._setup_ui()
        self._start_polling()

    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="System Monitor", font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.status_indicator = ctk.CTkLabel(header, text="Monitoring...", font=ctk.CTkFont(size=12), text_color="gray")
        self.status_indicator.grid(row=0, column=1, sticky="e")

        ctk.CTkButton(header, text="Refresh", width=80, command=self.refresh).grid(row=0, column=2, padx=5)
        ctk.CTkButton(header, text="Clear Logs", width=80, command=self._clear_logs).grid(row=0, column=3)

        # Left column - System metrics
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1)

        self._setup_system_metrics(left_frame)
        self._setup_inference_metrics(left_frame)
        self._setup_request_log(left_frame)

        # Right column - Gateway stats & jobs
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(2, weight=1)

        self._setup_gateway_stats(right_frame)
        self._setup_active_jobs(right_frame)
        self._setup_redis_queue(right_frame)

    def _setup_system_metrics(self, parent):
        ctk.CTkLabel(parent, text="System Resources", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=15, pady=(10, 5), sticky="w"
        )

        # Resource cards
        cards = ctk.CTkFrame(parent, fg_color="transparent")
        cards.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        cards.grid_columnconfigure((0, 1, 2), weight=1)

        # CPU
        self.cpu_card = self._create_resource_card(cards, "CPU", "0%", 0)
        # RAM
        self.ram_card = self._create_resource_card(cards, "RAM", "0/0 GB", 1)
        # GPU
        self.gpu_card = self._create_resource_card(cards, "GPU", "N/A", 2)

        # CPU bar
        bar_frame = ctk.CTkFrame(parent)
        bar_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        bar_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(bar_frame, text="CPU Usage:", width=100).grid(row=0, column=0, padx=5, sticky="w")
        self.cpu_progress = ctk.CTkProgressBar(bar_frame, height=15)
        self.cpu_progress.grid(row=0, column=1, padx=5, sticky="ew")
        self.cpu_progress.set(0)

        # RAM bar
        bar2_frame = ctk.CTkFrame(parent)
        bar2_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        bar2_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(bar2_frame, text="RAM Usage:", width=100).grid(row=0, column=0, padx=5, sticky="w")
        self.ram_progress = ctk.CTkProgressBar(bar2_frame, height=15)
        self.ram_progress.grid(row=0, column=1, padx=5, sticky="ew")
        self.ram_progress.set(0)

        # GPU bar
        bar3_frame = ctk.CTkFrame(parent)
        bar3_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        bar3_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(bar3_frame, text="GPU Usage:", width=100).grid(row=0, column=0, padx=5, sticky="w")
        self.gpu_progress = ctk.CTkProgressBar(bar3_frame, height=15)
        self.gpu_progress.grid(row=0, column=1, padx=5, sticky="ew")
        self.gpu_progress.set(0)
        self.gpu_label = ctk.CTkLabel(bar3_frame, text="N/A")
        self.gpu_label.grid(row=0, column=2, padx=5)

    def _create_resource_card(self, parent, title: str, value: str, col: int) -> dict:
        card = ctk.CTkFrame(parent)
        card.grid(row=0, column=col, padx=3, sticky="ew")

        label = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11), text_color="gray")
        label.pack(pady=(8, 2))

        value_label = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=16, weight="bold"))
        value_label.pack(pady=(0, 8))

        return {"card": card, "label": value_label}

    def _setup_inference_metrics(self, parent):
        ctk.CTkLabel(parent, text="Inference Statistics", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=5, column=0, padx=15, pady=(15, 5), sticky="w"
        )

        stats_frame = ctk.CTkFrame(parent)
        stats_frame.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        stats_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.total_requests = self._create_stat_box(stats_frame, "Total Requests", "0", 0)
        self.avg_latency = self._create_stat_box(stats_frame, "Avg Latency", "0ms", 1)
        self.tokens_generated = self._create_stat_box(stats_frame, "Tokens", "0", 2)

        # Request rate
        rate_frame = ctk.CTkFrame(parent)
        rate_frame.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
        rate_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(rate_frame, text="Request Rate:", width=100).grid(row=0, column=0, padx=5, sticky="w")
        self.rate_progress = ctk.CTkProgressBar(rate_frame, height=15)
        self.rate_progress.grid(row=0, column=1, padx=5, sticky="ew")
        self.rate_progress.set(0)
        self.rate_label = ctk.CTkLabel(rate_frame, text="0 req/min")
        self.rate_label.grid(row=0, column=2, padx=5)

    def _create_stat_box(self, parent, title: str, value: str, col: int) -> ctk.CTkLabel:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=col, padx=3, sticky="ew")

        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=10), text_color="gray").pack(pady=(5, 2))

        label = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=14, weight="bold"))
        label.pack(pady=(0, 5))

        return label

    def _setup_request_log(self, parent):
        ctk.CTkLabel(parent, text="Request Log", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=8, column=0, padx=15, pady=(15, 5), sticky="w"
        )

        log_frame = ctk.CTkFrame(parent)
        log_frame.grid(row=9, column=0, padx=10, pady=(0, 10), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        # Log header
        log_header = ctk.CTkFrame(log_frame, fg_color=("gray80", "gray25"))
        log_header.grid(row=0, column=0, sticky="ew")
        log_header.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(log_header, text="Time", font=ctk.CTkFont(size=10, weight="bold")).grid(
            row=0, column=0, padx=5, pady=2, sticky="w"
        )
        ctk.CTkLabel(log_header, text="Model", font=ctk.CTkFont(size=10, weight="bold")).grid(
            row=0, column=1, padx=5, pady=2, sticky="w"
        )
        ctk.CTkLabel(log_header, text="Latency", font=ctk.CTkFont(size=10, weight="bold")).grid(
            row=0, column=2, padx=5, pady=2, sticky="w"
        )
        ctk.CTkLabel(log_header, text="Status", font=ctk.CTkFont(size=10, weight="bold")).grid(
            row=0, column=3, padx=5, pady=2, sticky="w"
        )

        self.request_log = ctk.CTkScrollableFrame(log_frame, height=150)
        self.request_log.grid(row=1, column=0, sticky="nsew")
        self.request_log.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._request_entries = []

    def _setup_gateway_stats(self, parent):
        ctk.CTkLabel(parent, text="Gateway Statistics", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=15, pady=(10, 5), sticky="w"
        )

        stats_frame = ctk.CTkFrame(parent)
        stats_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        stats_frame.grid_columnconfigure((0, 1), weight=1)

        self.uptime_label = self._create_gateway_stat(stats_frame, "Uptime", "--", 0, 0)
        self.models_loaded_label = self._create_gateway_stat(stats_frame, "Models Loaded", "0", 0, 1)
        self.queue_size_label = self._create_gateway_stat(stats_frame, "Queue Size", "0", 1, 0)
        self.active_jobs_label = self._create_gateway_stat(stats_frame, "Active Jobs", "0", 1, 1)

    def _create_gateway_stat(self, parent, title: str, value: str, row: int, col: int) -> ctk.CTkLabel:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(5, 2))
        label = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(pady=(0, 5))

        return label

    def _setup_active_jobs(self, parent):
        ctk.CTkLabel(parent, text="Active Training Jobs", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=3, column=0, padx=15, pady=(15, 5), sticky="w"
        )

        jobs_frame = ctk.CTkScrollableFrame(parent)
        jobs_frame.grid(row=4, column=0, padx=10, pady=5, sticky="nsew")
        jobs_frame.grid_columnconfigure(0, weight=1)

        self.jobs_list = jobs_frame
        self._job_rows = []

        # Empty state
        self._no_jobs_label = ctk.CTkLabel(jobs_frame, text="No active jobs", text_color="gray")
        self._no_jobs_label.pack(pady=20)

    def _setup_redis_queue(self, parent):
        ctk.CTkLabel(parent, text="Redis Queue", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=5, column=0, padx=15, pady=(15, 5), sticky="w"
        )

        queue_frame = ctk.CTkFrame(parent)
        queue_frame.grid(row=6, column=0, padx=10, pady=(0, 10), sticky="ew")
        queue_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.queue_pending = self._create_queue_stat(queue_frame, "Pending", "0", 0)
        self.queue_active = self._create_queue_stat(queue_frame, "Active", "0", 1)
        self.queue_completed = self._create_queue_stat(queue_frame, "Completed", "0", 2)

    def _create_queue_stat(self, parent, title: str, value: str, col: int) -> ctk.CTkLabel:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=col, padx=3, sticky="ew")

        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=10), text_color="gray").pack(pady=(5, 2))

        label = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=14, weight="bold"))
        label.pack(pady=(0, 5))

        return label

    def _start_polling(self):
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll, daemon=True)
        self._poll_thread.start()

    def _poll(self):
        while self._polling:
            self._update_system_metrics()
            self._update_gateway_stats()
            self._update_job_list()
            time.sleep(2)

    def _update_system_metrics(self):
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()

            # RAM
            memory = psutil.virtual_memory()
            ram_used = memory.used / (1024**3)
            ram_total = memory.total / (1024**3)
            ram_percent = memory.percent

            # GPU (try to get if available)
            gpu_info = self._get_gpu_info()

            self.after(0, lambda: self.cpu_card["label"].configure(text=f"{cpu_percent:.0f}%"))
            self.after(0, lambda: self.ram_card["label"].configure(text=f"{ram_used:.1f}/{ram_total:.1f}GB"))
            self.after(0, lambda: self.cpu_progress.set(cpu_percent / 100))
            self.after(0, lambda: self.ram_progress.set(ram_percent / 100))

            if gpu_info:
                self.after(0, lambda: self.gpu_card["label"].configure(text=gpu_info["text"]))
                self.after(0, lambda: self.gpu_progress.set(gpu_info["percent"] / 100))
                self.after(0, lambda: self.gpu_label.configure(text=f"{gpu_info['memory']}"))
            else:
                self.after(0, lambda: self.gpu_card["label"].configure(text="N/A"))
                self.after(0, lambda: self.gpu_progress.set(0))
                self.after(0, lambda: self.gpu_label.configure(text="No GPU"))

        except Exception:
            pass

    def _get_gpu_info(self) -> Optional[dict]:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_id = 0
                mem_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(gpu_id) if device_count > 0 else "GPU"

                # Try to get utilization (may not work on all systems)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = util.gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem = f"{mem_info.used/(1024**3):.1f}/{mem_info.total/(1024**3):.1f}GB"
                    pynvml.nvmlShutdown()
                except ImportError:
                    gpu_percent = 0
                    gpu_mem = f"{mem_allocated:.1f}GB alloc"

                return {
                    "text": f"{device_name[:15]}",
                    "percent": gpu_percent,
                    "memory": gpu_mem,
                }
        except ImportError:
            pass
        return None

    def _update_gateway_stats(self):
        try:
            if not self.app.api.is_connected():
                self.after(0, lambda: self.status_indicator.configure(text="Disconnected", text_color="red"))
                return

            self.after(0, lambda: self.status_indicator.configure(text="Monitoring...", text_color="gray"))

            # Get metrics
            try:
                metrics = self.app.api.get_metrics()
                uptime = metrics.get("uptime_seconds", 0)
                uptime_str = self._format_uptime(uptime)
                self.after(0, lambda: self.uptime_label.configure(text=uptime_str))
            except Exception:
                self.after(0, lambda: self.uptime_label.configure(text="Error"))

            # Get runtime info
            try:
                runtime = self.app.api.runtime_health()
                loaded_count = 0
                for backend, info in runtime.items():
                    loaded_count += len(info.get("loaded_models", []))
                self.after(0, lambda: self.models_loaded_label.configure(text=str(loaded_count)))
            except Exception:
                self.after(0, lambda: self.models_loaded_label.configure(text="?"))

            # Get training jobs
            try:
                jobs = self.app.api.list_training_jobs()
                active = len([j for j in jobs if j.get("status") == "running"])
                pending = len([j for j in jobs if j.get("status") in ("queued", "pending")])
                completed = len([j for j in jobs if j.get("status") == "completed"])

                self.after(0, lambda: self.active_jobs_label.configure(text=str(active)))
                self.after(0, lambda: self.queue_pending.configure(text=str(pending)))
                self.after(0, lambda: self.queue_active.configure(text=str(active)))
                self.after(0, lambda: self.queue_completed.configure(text=str(completed)))
            except Exception:
                pass

        except Exception:
            pass

    def _format_uptime(self, seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        else:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            return f"{hours}h {mins}m"

    def _update_job_list(self):
        try:
            jobs = self.app.api.list_training_jobs()
            active_jobs = [j for j in jobs if j.get("status") in ("running", "queued")]

            # Clear existing rows
            for row in self._job_rows:
                row.destroy()
            self._job_rows.clear()

            if not active_jobs:
                self._no_jobs_label.pack(pady=20)
            else:
                self._no_jobs_label.pack_forget()

            for job in active_jobs:
                self._create_job_row(job)

        except Exception:
            pass

    def _create_job_row(self, job: dict):
        row = ctk.CTkFrame(self.jobs_list)
        row.pack(fill="x", pady=2)

        job_id = job.get("job_id", "?")[:20]
        config = job.get("config", {})
        output = config.get("output_name", job_id)
        status = job.get("status", "unknown")
        progress = job.get("progress", 0)

        status_colors = {
            "running": "blue",
            "queued": "orange",
            "completed": "green",
            "failed": "red",
        }

        ctk.CTkLabel(row, text=output[:25], anchor="w", font=ctk.CTkFont(size=11)).grid(
            row=0, column=0, padx=5, pady=3, sticky="w"
        )
        ctk.CTkLabel(
            row, text=status, text_color=status_colors.get(status, "gray"), font=ctk.CTkFont(size=11)
        ).grid(row=0, column=1, padx=5, pady=3, sticky="e")

        # Progress bar
        progress_bar = ctk.CTkProgressBar(row, height=8)
        progress_bar.grid(row=1, column=0, columnspan=2, padx=5, pady=(0, 3), sticky="ew")
        progress_bar.set(progress / 100)

        self._job_rows.append(row)

    def _add_request_entry(self, model: str, latency: float, status: str):
        timestamp = time.strftime("%H:%M:%S")

        entry = ctk.CTkFrame(self.request_log)
        entry.pack(fill="x", pady=1)

        colors = {"success": "green", "error": "red", "streaming": "blue"}
        status_color = colors.get(status, "gray")

        ctk.CTkLabel(entry, text=timestamp, font=ctk.CTkFont(size=10), text_color="gray").grid(
            row=0, column=0, padx=5, pady=2, sticky="w"
        )
        ctk.CTkLabel(entry, text=model[:20], font=ctk.CTkFont(size=10), anchor="w").grid(
            row=0, column=1, padx=5, pady=2, sticky="w"
        )
        ctk.CTkLabel(entry, text=f"{latency:.0f}ms", font=ctk.CTkFont(size=10)).grid(
            row=0, column=2, padx=5, pady=2, sticky="w"
        )
        ctk.CTkLabel(entry, text=status, text_color=status_color, font=ctk.CTkFont(size=10)).grid(
            row=0, column=3, padx=5, pady=2, sticky="e"
        )

        self._request_entries.append(entry)

        # Keep only last 50 entries
        if len(self._request_entries) > 50:
            old = self._request_entries.pop(0)
            old.destroy()

    def _clear_logs(self):
        for entry in self._request_entries:
            entry.destroy()
        self._request_entries.clear()

    def refresh(self):
        self._update_system_metrics()
        self._update_gateway_stats()
        self._update_job_list()

    def destroy(self):
        self._polling = False
        super().destroy()
