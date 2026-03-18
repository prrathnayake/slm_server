# Contributing to Local LLM Platform

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/local-llm-platform.git
   cd local-llm-platform
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
4. Install in development mode:
   ```bash
   pip install -e ".[all]"
   ```
5. Copy the environment file:
   ```bash
   cp .env.example .env
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   python -m pytest
   ```
4. Run linting:
   ```bash
   black local_llm_platform/
   isort local_llm_platform/
   ruff check local_llm_platform/
   ```
5. Commit your changes:
   ```bash
   git commit -m "feat: description of your change"
   ```
6. Push and create a pull request

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add remote HTTP runtime backend
fix: resolve streaming timeout issue
docs: update installation instructions
```

## Code Style

- Use Black for formatting
- Use isort for import sorting
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write docstrings for public functions and classes

## Project Structure

When adding new features, follow the existing structure:

- **Apps** (`apps/`) - Application entry points (API servers, CLI, UI)
- **Core** (`core/`) - Shared configuration, logging, security
- **Services** (`services/`) - Business logic services
- **Runtimes** (`runtimes/`) - Model inference backends
- **Training** (`training/`) - Fine-tuning pipelines

## Adding a New Runtime Backend

1. Create a new file in `runtimes/` (e.g., `my_runtime.py`)
2. Inherit from `BaseRuntime` in `runtimes/base.py`
3. Implement required methods: `load_model`, `unload_model`, `generate`
4. Register the runtime in the runtime manager
5. Add tests in `tests/runtimes/`

## Adding a New Training Pipeline

1. Create a new file in `training/pipelines/`
2. Inherit from `BaseTrainer` in `training/base.py`
3. Implement required methods: `train`, `validate`, `export`
4. Register the pipeline in the trainer worker
5. Add tests in `tests/training/`

## Testing

- Write unit tests for new functionality
- Ensure existing tests pass before submitting PR
- Use pytest fixtures for common setup
- Mock external services (Redis, model backends)

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_registry.py

# Run with coverage
python -m pytest --cov=local_llm_platform
```

## Documentation

- Update README.md if adding user-facing features
- Add docstrings to new functions and classes
- Update CHANGELOG.md with your changes
- Add API documentation for new endpoints

## Reporting Issues

- Use the GitHub issue tracker
- Include steps to reproduce bugs
- Include your environment details (OS, Python version)
- Include relevant logs or error messages

## Questions?

Open a GitHub issue for questions or discussions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
