import customtkinter as ctk
import threading
import time
from tkinter import filedialog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_llm_platform.apps.desktop.app import SLMApp


class TrainingPage(ctk.CTkFrame):
    """Training page - create and monitor training jobs."""

    def __init__(self, master, app: "SLMApp", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self._polling = False
        self._active_job_id = None

        # Title
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="Training", font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, sticky="w"
        )

        ctk.CTkButton(header, text="Refresh Jobs", width=120, command=self.refresh_jobs).grid(
            row=0, column=1, sticky="e"
        )

        # Main content - split into config (left) and jobs (right)
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Left: Training config
        config_frame = ctk.CTkFrame(content)
        config_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        config_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(config_frame, text="New Training Job", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w"
        )

        self.fields = {}
        row = 1

        labels = [
            ("base_model", "Base Model:", "entry"),
            ("dataset_id", "Dataset ID:", "entry"),
            ("output_name", "Output Name:", "entry"),
            ("training_type", "Type:", "combo", ["lora", "qlora", "sft"]),
            ("epochs", "Epochs:", "entry"),
            ("batch_size", "Batch Size:", "entry"),
            ("learning_rate", "Learning Rate:", "entry"),
            ("lora_r", "LoRA Rank (r):", "entry"),
            ("lora_alpha", "LoRA Alpha:", "entry"),
            ("max_seq_length", "Max Seq Length:", "entry"),
            ("target", "Execution:", "combo", ["local", "remote"]),
        ]

        for label_info in labels:
            field_id = label_info[0]
            label_text = label_info[1]
            field_type = label_info[2]

            ctk.CTkLabel(config_frame, text=label_text).grid(
                row=row, column=0, padx=15, pady=3, sticky="w"
            )

            if field_type == "entry":
                widget = ctk.CTkEntry(config_frame, width=200)
                widget.grid(row=row, column=1, padx=15, pady=3, sticky="ew")
            elif field_type == "combo":
                widget = ctk.CTkComboBox(config_frame, values=label_info[3], width=200)
                widget.grid(row=row, column=1, padx=15, pady=3, sticky="ew")
                widget.set(label_info[3][0])

            self.fields[field_id] = widget
            row += 1

        # Set defaults
        self.fields["epochs"].insert(0, "3")
        self.fields["batch_size"].insert(0, "4")
        self.fields["learning_rate"].insert(0, "2e-4")
        self.fields["lora_r"].insert(0, "16")
        self.fields["lora_alpha"].insert(0, "32")
        self.fields["max_seq_length"].insert(0, "2048")

        # Start button
        ctk.CTkButton(
            config_frame,
            text="Start Training",
            command=self._start_training,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=row, column=0, columnspan=2, padx=15, pady=15, sticky="ew")

        # Right: Jobs list and progress
        jobs_frame = ctk.CTkFrame(content)
        jobs_frame.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
        jobs_frame.grid_columnconfigure(0, weight=1)
        jobs_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(jobs_frame, text="Training Jobs", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=15, pady=(15, 10), sticky="w"
        )

        self.jobs_list = ctk.CTkScrollableFrame(jobs_frame)
        self.jobs_list.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.jobs_list.grid_columnconfigure(0, weight=1)

        # Progress section
        progress_frame = ctk.CTkFrame(jobs_frame)
        progress_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        progress_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(progress_frame, text="Progress", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=0, padx=10, pady=(5, 0), sticky="w"
        )
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(progress_frame, text="No active job")
        self.progress_label.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")

        # Training logs
        ctk.CTkLabel(progress_frame, text="Logs", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=3, column=0, padx=10, pady=(5, 0), sticky="w"
        )
        self.log_text = ctk.CTkTextbox(progress_frame, height=120)
        self.log_text.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _start_training(self):
        try:
            config = {
                "base_model": self.fields["base_model"].get(),
                "dataset_id": self.fields["dataset_id"].get(),
                "output_name": self.fields["output_name"].get(),
                "training_type": self.fields["training_type"].get(),
                "epochs": int(self.fields["epochs"].get() or "3"),
                "batch_size": int(self.fields["batch_size"].get() or "4"),
                "learning_rate": float(self.fields["learning_rate"].get() or "2e-4"),
                "lora_r": int(self.fields["lora_r"].get() or "16"),
                "lora_alpha": int(self.fields["lora_alpha"].get() or "32"),
                "max_seq_length": int(self.fields["max_seq_length"].get() or "2048"),
            }
            target = self.fields["target"].get()

            if not config["base_model"] or not config["output_name"]:
                self._append_log("Error: Base model and output name are required")
                return

            threading.Thread(
                target=lambda: self._do_start_training(config, target), daemon=True
            ).start()
        except Exception as e:
            self._append_log(f"Error: {e}")

    def _do_start_training(self, config, target):
        try:
            self._append_log(f"Starting training: {config['output_name']}...")
            result = self.app.api.create_training_job(config, target)
            job_id = result.get("job_id", "unknown")
            self._append_log(f"Job created: {job_id}")
            self._active_job_id = job_id
            self._start_polling(job_id)
            self.refresh_jobs()
        except Exception as e:
            self._append_log(f"Error starting training: {e}")

    def _start_polling(self, job_id):
        self._polling = True
        threading.Thread(target=lambda: self._poll_job(job_id), daemon=True).start()

    def _poll_job(self, job_id):
        while self._polling:
            try:
                progress = self.app.api.get_training_progress(job_id)
                status = progress.get("status", "unknown")
                pct = progress.get("progress", 0)
                loss = progress.get("loss")
                epoch = progress.get("current_epoch", 0)

                self.after(0, lambda p=pct: self.progress_bar.set(p / 100))
                label = f"Job {job_id}: {status} - {pct:.1f}% (epoch {epoch})"
                if loss:
                    label += f" - loss: {loss:.4f}"
                self.after(0, lambda l=label: self.progress_label.configure(text=l))

                logs = self.app.api.get_training_logs(job_id)
                if logs:
                    self.log_text.delete("1.0", "end")
                    for line in logs[-20:]:
                        self.log_text.insert("end", f"{line}\n")
                    self.log_text.see("end")

                if status in ("completed", "failed", "cancelled"):
                    self._polling = False
                    self.after(0, lambda: self._append_log(f"Job {job_id} {status}"))
                    self.after(0, self.refresh_jobs)
                    break

                time.sleep(2)
            except Exception:
                time.sleep(5)

    def refresh_jobs(self):
        threading.Thread(target=self._load_jobs, daemon=True).start()

    def _load_jobs(self):
        try:
            jobs = self.app.api.list_training_jobs()
            self.after(0, lambda: self._populate_jobs(jobs))
        except Exception as e:
            self._append_log(f"Error loading jobs: {e}")

    def _populate_jobs(self, jobs):
        for w in self.jobs_list.winfo_children():
            w.destroy()

        if not jobs:
            ctk.CTkLabel(self.jobs_list, text="No training jobs", text_color="gray").pack(pady=20)
            return

        for job in jobs:
            row = ctk.CTkFrame(self.jobs_list)
            row.pack(fill="x", pady=2)

            status = job.get("status", "unknown")
            status_color = {
                "running": "blue",
                "completed": "green",
                "failed": "red",
                "queued": "orange",
                "cancelled": "gray",
            }.get(status, "gray")

            info = f"{job.get('job_id', '?')}"
            config = job.get("config", {})
            if isinstance(config, dict):
                info += f"  |  {config.get('output_name', '?')}"

            ctk.CTkLabel(row, text=info, anchor="w").pack(side="left", padx=10, pady=5)
            ctk.CTkLabel(row, text=status, text_color=status_color).pack(side="right", padx=10, pady=5)

            job_id = job.get("job_id", "")
            if status in ("running", "queued"):
                ctk.CTkButton(
                    row, text="Cancel", width=60, height=24,
                    fg_color="orange", hover_color="darkorange",
                    command=lambda j=job_id: self._cancel_job(j),
                ).pack(side="right", padx=5, pady=5)

    def _cancel_job(self, job_id):
        threading.Thread(
            target=lambda: self._do_cancel(job_id), daemon=True
        ).start()

    def _do_cancel(self, job_id):
        try:
            self.app.api.cancel_training_job(job_id)
            self._append_log(f"Cancelled job {job_id}")
            self.refresh_jobs()
        except Exception as e:
            self._append_log(f"Error cancelling {job_id}: {e}")

    def _append_log(self, msg: str):
        def _do():
            self.log_text.insert("end", f"{msg}\n")
            self.log_text.see("end")

        self.after(0, _do)
