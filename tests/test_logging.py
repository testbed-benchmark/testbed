"""Tests for logging helpers."""

import json

from testbed.core.logging import JSONLLogger


class _FlakyFlush:
    def __init__(self, f):
        self.f = f
        self.flush_calls = 0

    def write(self, *args, **kwargs):
        return self.f.write(*args, **kwargs)

    def flush(self):
        self.flush_calls += 1
        if self.flush_calls == 1:
            raise BlockingIOError(11, "write could not complete without blocking")
        return self.f.flush()

    def close(self):
        return self.f.close()


def test_jsonl_logger_defers_temporary_flush_error_until_close(tmp_path):
    logger = JSONLLogger(tmp_path / "metrics.jsonl", close_retry_delay=0.0)
    flaky = _FlakyFlush(logger.f)
    logger.f = flaky

    logger.log({"step": 1, "episodic_return": 2.0})
    assert flaky.flush_calls == 1
    assert logger.pending_flush

    logger.close()

    assert flaky.flush_calls == 2
    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert record["step"] == 1
    assert record["episodic_return"] == 2.0
