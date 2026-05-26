"""Logging helpers for experiments."""

import json
import logging
import time


class BaseLogger:
    """Base interface for metric loggers."""

    def log(self, key: str, value: float, step: int):
        """Logs one metric."""
        ...

    def log_dict(self, d: dict, step: int):
        """Logs multiple metrics."""
        ...

    def close(self):
        """Closes logger resources."""
        ...


class PrintLogger(BaseLogger):
    """Prints metrics to stdout."""

    def log(self, metrics: dict, step: int):
        """Prints a metrics dictionary."""
        msg = f"[{step}] " + " ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        print(msg)

    def close(self):
        """Closes the logger."""
        pass


class EventLogger:
    """Thin wrapper around Python's stdlib logging for narrative/event logs.

    Writes to a .log file and optionally to console.
    """

    def __init__(self, path, console: bool = True, level=logging.DEBUG):
        """Initializes file and optional console logging."""
        self._logger = logging.getLogger(str(path))  # unique name per experiment
        self._logger.setLevel(level)
        self._logger.propagate = False  # don't bubble up to root logger

        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
        )

        # file handler — always on
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        # console handler — optional
        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

    # mirror the stdlib API so it feels natural
    def info(self, msg, *args, **kwargs):
        """Logs an info message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Logs a warning message."""
        self._logger.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Logs a debug message."""
        self._logger.debug(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Logs an error message."""
        self._logger.error(msg, *args, **kwargs)


class JSONLLogger:
    """Logs structured metric dicts as JSONL; good for plotting/analysis."""

    def __init__(self, path):
        """Opens the JSONL log file."""
        self.f = open(path, "a")
        self.start_time = time.perf_counter()

    def log(self, metrics: dict):
        """Appends one metrics record."""
        record = {
            "time": time.time() - self.start_time,
            **self._to_python(metrics),
        }
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()  # safer for long runs

    def close(self):
        """Closes the log file."""
        self.f.close()

    def _to_python(self, metrics):
        out = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):  # torch tensor
                out[k] = v.item()
            else:
                out[k] = v
        return out
