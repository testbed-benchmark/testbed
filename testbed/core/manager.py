"""Experiment run management."""

import os
import time

import yaml

from testbed.core.logging import (
    EventLogger,
    JSONLLogger,
    MultiLogger,
    WandbLogger,
    config_to_dict,
)


class Manager:
    """Manage run directories, config snapshots, and loggers."""

    def __init__(
        self,
        cfg,
        root="runs",
        name=None,
        logger=None,
        console=True,
        timestamp=True,
        run_dir=None,
    ):
        """Create a run directory and loggers.

        Args:
            cfg: Experiment configuration to save with the run and pass to
                configured loggers.
            root: Parent directory in which to create the run directory.
            name: Run name. If omitted, the current timestamp is used.
            logger: Metric logger to use. If omitted, a JSONL logger is created,
                optionally alongside a Weights & Biases logger configured by
                ``cfg.wandb``.
            console: Whether to write event log messages to the console.
            timestamp: Whether to prefix ``name`` with the current timestamp.
                Ignored when ``name`` is omitted or ``run_dir`` is provided.
            run_dir: Existing run directory to adopt instead of constructing the
                default ``root/[timestamp_]name``.
        """
        self.cfg = cfg

        # create run directory
        if run_dir is not None:
            self.dir = run_dir
            self.run_name = os.path.basename(os.path.normpath(run_dir))
        else:
            timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            if name is not None:
                self.run_name = f"{timestamp_str}_{name}" if timestamp else name
            else:
                self.run_name = timestamp_str
            self.dir = os.path.join(root, self.run_name)
        os.makedirs(self.dir, exist_ok=True)

        # save config
        self.config_path = os.path.join(self.dir, "config.yaml")
        self._save_config()

        # event logging (create first so metric-logger setup can report issues)
        events_path = os.path.join(self.dir, "events.log")
        self.event_logger = EventLogger(events_path, console=console)

        self.event_logger.info("Spawning Manager...")
        self.event_logger.info(f"Run directory: {self.dir}")

        # metric logging
        if logger is not None:
            self.metric_logger = logger
        else:
            metrics_path = os.path.join(self.dir, "metrics.jsonl")
            jsonl_logger = JSONLLogger(metrics_path)
            self.metric_logger = jsonl_logger

            wandb_cfg = self._wandb_cfg()
            if wandb_cfg.get("enabled", False):
                # Never let a wandb backend failure (missing dependency, auth,
                # network, quota) take down the training run: warn and fall back
                # to the JSONL logger that's already in place.
                try:
                    wandb_logger = WandbLogger(
                        cfg,
                        run_name=self.run_name,
                        project=wandb_cfg.get("project", "testbed"),
                        entity=wandb_cfg.get("entity"),
                        tags=wandb_cfg.get("tags"),
                        mode=wandb_cfg.get("mode", "online"),
                        group=wandb_cfg.get("group"),
                    )
                except Exception as e:  # noqa: BLE001 - backend failures are non-fatal
                    self.event_logger.warning(
                        f"wandb logging is enabled but failed to initialize "
                        f"({e!r}); continuing with JSONL logging only."
                    )
                else:
                    self.metric_logger = MultiLogger([jsonl_logger, wandb_logger])

    def _save_config(self):
        """Saves the experiment configuration.

        Config types supported: DataClass, OmegaConf, Dict, etc.
        """
        with open(self.config_path, "w") as f:
            yaml.dump(config_to_dict(self.cfg), f)

    def _wandb_cfg(self) -> dict:
        """Reads the optional ``wandb`` config block as a plain dict.

        Returns an empty dict when the config has no ``wandb`` section, so wandb
        stays off by default. Runs the block through the same conversion path as
        the rest of the config, so an object-style block (namespace/dataclass) is
        handled rather than assumed to already be a mapping.
        """
        block = config_to_dict(self.cfg).get("wandb")
        if block is None:
            return {}
        return config_to_dict(block)

    def path(self, *parts):
        """Builds paths inside the run directory."""
        return os.path.join(self.dir, *parts)
