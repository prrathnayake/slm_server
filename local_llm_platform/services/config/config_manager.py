from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.config")


class ConfigManager:
    """Manages runtime configuration that can be changed without restart."""

    DEFAULT_CONFIG = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
        },
        "models": {
            "default_backend": "llama_cpp",
            "max_loaded": 3,
            "load_timeout": 300,
            "auto_discover": True,
        },
        "training": {
            "max_concurrent_jobs": 2,
            "default_epochs": 3,
            "default_batch_size": 4,
            "default_lr": "2e-4",
        },
        "security": {
            "auth_enabled": False,
            "rate_limit": 100,
        },
        "pool": {
            "hot_models": [],
            "eviction_policy": "lru",
        },
    }

    def __init__(self, config_path: str = "./config.json"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    self._config = json.load(f)
                logger.info("Loaded config from file")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
                self._config = copy.deepcopy(self.DEFAULT_CONFIG)
        else:
            self._config = copy.deepcopy(self.DEFAULT_CONFIG)
            self._save()

    def _save(self) -> None:
        with open(self.config_path, "w") as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save()
        logger.info(f"Config updated: {key} = {value}")

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})

    def get_all(self) -> Dict[str, Any]:
        return self._config.copy()

    def reset(self) -> None:
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)
        self._save()
        logger.info("Config reset to defaults")
