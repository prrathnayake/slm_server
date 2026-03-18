#!/usr/bin/env python3
"""CLI admin tool for Local LLM Platform."""

import argparse
import asyncio
import json
import sys
from pathlib import Path


def cmd_models_list(args):
    """List all registered models."""
    from local_llm_platform.services.registry.registry import ModelRegistry
    registry = ModelRegistry()
    models = registry.list_models()
    for m in models:
        status_icon = "✓" if m.status.value == "ready" else "○"
        print(f"  {status_icon} {m.model_id:30s} [{m.runtime_backend.value:12s}] {m.status.value}")


def cmd_models_register(args):
    """Register a new model."""
    from local_llm_platform.services.registry.registry import ModelRegistry
    from local_llm_platform.core.schemas.models import (
        ModelRegistryEntry, ModelStatus, ModelFormat, BackendType, SourceType, Specialization
    )

    registry = ModelRegistry()
    entry = ModelRegistryEntry(
        model_id=args.model_id,
        display_name=args.name or args.model_id,
        runtime_backend=BackendType(args.backend),
        model_format=ModelFormat(args.format),
        status=ModelStatus.READY if args.ready else ModelStatus.UNLOADED,
        source=SourceType.IMPORTED_ZIP,
        specialization=Specialization(args.specialization) if args.specialization else Specialization.GENERAL,
        artifact_path=args.path,
    )
    result = registry.register(entry)
    print(f"Registered: {result.model_id}")


def cmd_models_unregister(args):
    """Unregister a model."""
    from local_llm_platform.services.registry.registry import ModelRegistry
    registry = ModelRegistry()
    if registry.unregister(args.model_id):
        print(f"Unregistered: {args.model_id}")
    else:
        print(f"Model not found: {args.model_id}")
        sys.exit(1)


def cmd_import(args):
    """Import a model zip file."""
    async def do_import():
        from local_llm_platform.training.import_trainer import ImportProcessor
        processor = ImportProcessor()
        result = await processor.process_zip(args.file, args.model_id)
        print(json.dumps(result, indent=2, default=str))

    asyncio.run(do_import())


def cmd_status(args):
    """Show platform status."""
    from local_llm_platform.core.config.settings import settings
    from local_llm_platform.services.registry.registry import ModelRegistry

    registry = ModelRegistry()
    models = registry.list_models()
    ready = len([m for m in models if m.status.value == "ready"])
    total = len(models)

    print(f"Local LLM Platform v{settings.APP_VERSION}")
    print(f"Models: {ready}/{total} ready")
    print(f"Database: {settings.DATABASE_URL}")
    print(f"Models dir: {settings.MODELS_DIR}")
    print(f"Datasets dir: {settings.DATASETS_DIR}")


def cmd_serve(args):
    """Start the gateway API server."""
    from local_llm_platform.apps.gateway_api.main import start
    start()


def cmd_ui(args):
    """Launch the desktop UI application."""
    from local_llm_platform.apps.desktop.app import main as ui_main
    ui_main()


def main():
    parser = argparse.ArgumentParser(
        prog="llp",
        description="Local LLM Platform CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the gateway API server")

    # ui
    ui_parser = subparsers.add_parser("ui", help="Launch desktop UI application")

    # status
    status_parser = subparsers.add_parser("status", help="Show platform status")

    # models
    models_parser = subparsers.add_parser("models", help="Model management")
    models_sub = models_parser.add_subparsers(dest="models_command")

    models_list = models_sub.add_parser("list", help="List all models")

    models_reg = models_sub.add_parser("register", help="Register a model")
    models_reg.add_argument("model_id", help="Model ID")
    models_reg.add_argument("--name", help="Display name")
    models_reg.add_argument("--backend", default="llama_cpp", choices=["llama_cpp", "vllm", "tgi", "remote_http", "remote_ssh"])
    models_reg.add_argument("--format", default="gguf", choices=["gguf", "safetensors", "adapter"])
    models_reg.add_argument("--path", help="Artifact path")
    models_reg.add_argument("--specialization", choices=["reasoning", "planning", "tool-calling", "personality", "coder", "general"])
    models_reg.add_argument("--ready", action="store_true", help="Mark as ready")

    models_unreg = models_sub.add_parser("unregister", help="Unregister a model")
    models_unreg.add_argument("model_id", help="Model ID")

    # import
    import_parser = subparsers.add_parser("import", help="Import a model zip")
    import_parser.add_argument("file", help="Path to zip file")
    import_parser.add_argument("--model-id", help="Target model ID")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "ui":
        cmd_ui(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "models":
        if args.models_command == "list":
            cmd_models_list(args)
        elif args.models_command == "register":
            cmd_models_register(args)
        elif args.models_command == "unregister":
            cmd_models_unregister(args)
        else:
            models_parser.print_help()
    elif args.command == "import":
        cmd_import(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
