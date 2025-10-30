import logging
import os
import unittest
from unittest.mock import patch

from src.lib.diagnostics.memwatch import MemoryWatchConfig, MemoryWatchdog


class MemoryWatchConfigTests(unittest.TestCase):
    def test_config_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MEM_WATCH": "1",
                "MEM_WATCH_INTERVAL": "5",
                "MEM_WATCH_WARMUP": "0",
                "MEM_WATCH_WARN_MB": "512",
                "MEM_WATCH_CRITICAL_MB": "1024",
                "MEM_WATCH_GC": "1",
                "MEM_WATCH_LOG_LEVEL": str(logging.DEBUG),
            },
            clear=True,
        ):
            config = MemoryWatchConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertEqual(config.interval_seconds, 5.0)
        self.assertEqual(config.warmup_seconds, 0.0)
        self.assertEqual(config.warn_threshold_bytes, 512 * 1024 * 1024)
        self.assertEqual(config.critical_threshold_bytes, 1024 * 1024 * 1024)
        self.assertTrue(config.gc_on_warning)
        self.assertEqual(config.log_level, logging.DEBUG)

    def test_config_disabled_when_interval_zero(self) -> None:
        with patch.dict(os.environ, {"MEM_WATCH": "1", "MEM_WATCH_INTERVAL": "0"}, clear=True):
            config = MemoryWatchConfig.from_env()

        self.assertFalse(config.enabled)


class MemoryWatchdogTests(unittest.TestCase):
    def test_warning_logs_when_threshold_exceeded(self) -> None:
        values = [100 * 1024 * 1024, 600 * 1024 * 1024]

        def sampler() -> int:
            return values.pop(0)

        config = MemoryWatchConfig(
            enabled=True,
            interval_seconds=1.0,
            warmup_seconds=0.0,
            warn_threshold_bytes=512 * 1024 * 1024,
            critical_threshold_bytes=0,
            gc_on_warning=False,
            log_level=logging.INFO,
            emit_debug_delta=True,
        )

        watchdog = MemoryWatchdog(config, sampler=sampler)

        with self.assertLogs("diag.memwatch", level="DEBUG") as captured:
            watchdog._sample()
            watchdog._sample()

        self.assertTrue(any("memory_usage_warning" in entry for entry in captured.output))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

