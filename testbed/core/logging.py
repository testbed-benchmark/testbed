"""Logging helpers for experiments."""

import json
import logging
import time
from pathlib import Path


class BaseLogger:
    """Base interface for metric loggers."""

    def log(self, key: str, value: float, step: int):
        """Logs one metric."""

    def log_dict(self, d: dict, step: int):
        """Logs multiple metrics."""

    def close(self):
        """Closes logger resources."""


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

    def __init__(self, path, close_retries: int = 5, close_retry_delay: float = 0.1):
        """Opens the JSONL log file."""
        self.f = Path(path).open(  # noqa: SIM115 - logger.close manages this long-lived handle
            "a"
        )
        self.start_time = time.perf_counter()
        self.close_retries = close_retries
        self.close_retry_delay = close_retry_delay
        self.pending_flush = False

    def log(self, metrics: dict):
        """Appends one metrics record."""
        record = {
            # Elapsed seconds since the logger was opened. Must use the same
            # clock as ``start_time`` (perf_counter), not wall-clock time.time().
            "time": time.perf_counter() - self.start_time,
            **_to_python(metrics),
        }
        self.f.write(json.dumps(record) + "\n")
        try:
            self.f.flush()
            self.pending_flush = False
        except BlockingIOError:
            self.pending_flush = True

    def close(self):
        """Closes the log file."""
        self._flush_before_close()
        self.f.close()

    def _flush_before_close(self):
        """Flushes pending metrics before closing."""
        for attempt in range(self.close_retries + 1):
            try:
                self.f.flush()
                self.pending_flush = False
                return
            except BlockingIOError:
                if attempt == self.close_retries:
                    raise
                time.sleep(self.close_retry_delay * 2**attempt)


class WandbLogger(BaseLogger):
    """Logs metric dicts to Weights & Biases.

    Implements the same ``log(dict)`` / ``close()`` protocol as
    :class:`JSONLLogger`, so it can be dropped in as a ``Manager`` metric logger
    without touching the trainers. ``wandb`` is imported lazily so this module
    stays importable without the optional dependency installed.
    """

    def __init__(
        self,
        cfg,
        run_name,
        project,
        entity=None,
        tags=None,
        mode="online",
        group=None,
    ):
        """Initializes a wandb run.

        Args:
            cfg: The experiment config (OmegaConf, dataclass, or dict). Snapshotted
                as the wandb run config.
            run_name: Display name for the wandb run.
            project: wandb project to log to.
            entity: wandb team/user; ``None`` uses the default.
            tags: Optional list of run tags.
            mode: One of ``"online"``, ``"offline"``, ``"disabled"``.
            group: Optional wandb run group.
        """
        try:
            import wandb
        except ImportError as e:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "wandb is required for WandbLogger; install the extra with "
                '`pip install -e ".[wandb]"`.'
            ) from e

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=list(tags) if tags is not None else None,
            group=group,
            mode=mode,
            config=config_to_dict(cfg),
        )

    def log(self, metrics: dict):
        """Logs one metrics dict, using its ``step`` for the wandb x-axis."""
        payload = _to_python(metrics)
        step = payload.pop("step", None)
        self.run.log(payload, step=step)

    def close(self):
        """Finishes the wandb run."""
        self.run.finish()


class MultiLogger(BaseLogger):
    """Fans a metric dict out to several loggers (e.g. JSONL + wandb)."""

    def __init__(self, loggers: list):
        """Stores the child loggers."""
        self.loggers = loggers

    def log(self, metrics: dict):
        """Forwards a copy of the metrics to each child logger."""
        for logger in self.loggers:
            logger.log(dict(metrics))

    def close(self):
        """Closes every child logger."""
        for logger in self.loggers:
            logger.close()


def _to_python(metrics: dict) -> dict:
    """Coerces torch tensors in a metrics dict to Python scalars."""
    out = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):  # torch tensor
            out[k] = v.item()
        else:
            out[k] = v
    return out


def config_to_dict(cfg):
    """Best-effort conversion of a config object to a plain dict.

    Handles OmegaConf configs, dataclasses, and plain dicts. Used both for
    snapshotting the config to disk and for the wandb run config.
    """
    try:
        from omegaconf import OmegaConf
        from omegaconf.errors import OmegaConfBaseException

        is_config = OmegaConf.is_config(cfg)
    except ImportError:
        is_config = False

    if is_config:
        try:
            return OmegaConf.to_container(cfg, resolve=True)
        except OmegaConfBaseException:
            # e.g. an unresolvable interpolation. Fall back to the raw
            # (unresolved) container rather than leaking OmegaConf internals
            # via vars(cfg).
            return OmegaConf.to_container(cfg, resolve=False)

    from dataclasses import asdict, is_dataclass

    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return cfg
    return vars(cfg)
