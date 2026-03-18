import pytest

from local_llm_platform.core.logging.logger import setup_logging, get_logger, JSONFormatter
from local_llm_platform.core.security.auth import generate_api_key, hash_api_key


class TestLogging:
    def test_get_logger(self):
        logger = get_logger("test")
        assert logger.name == "local_llm_platform.test"

    def test_json_formatter(self):
        import logging
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        import json
        parsed = json.loads(output)
        assert parsed["message"] == "test message"
        assert parsed["level"] == "INFO"


class TestSecurity:
    def test_generate_api_key(self):
        key = generate_api_key()
        assert key.startswith("llp-")
        assert len(key) > 10

    def test_generate_unique_keys(self):
        keys = [generate_api_key() for _ in range(10)]
        assert len(set(keys)) == 10

    def test_hash_api_key(self):
        hash1 = hash_api_key("test-key")
        hash2 = hash_api_key("test-key")
        hash3 = hash_api_key("different-key")
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64
