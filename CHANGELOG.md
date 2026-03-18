# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- FastAPI gateway with OpenAI-compatible endpoints
- Model registry with SQLite backend
- Runtime manager for model lifecycle
- Trainer worker for fine-tuning jobs
- llama.cpp runtime backend
- Local LoRA/QLoRA training pipeline
- SSE streaming support
- Admin CLI tool
- Desktop UI (optional)
- Docker and Docker Compose support
- Redis-based job queue
- Structured logging with structlog

## [0.1.0] - 2026-03-19

### Added
- Initial release
- Core platform architecture
- OpenAI-compatible API gateway (`/v1/models`, `/v1/chat/completions`, `/v1/completions`)
- Model registry and discovery
- Hot/cold model loading
- Fine-tuning with PEFT and TRL
- Model import from zip archives
- CLI administration tool
- Desktop GUI with CustomTkinter
- Docker deployment support
