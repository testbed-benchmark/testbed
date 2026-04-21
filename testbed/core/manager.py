import os
import time
import yaml

from testbed.core.logging import JSONLLogger, EventLogger


class Manager:
    def __init__(self, cfg, root="runs", name=None, logger=None, console=True, timestamp=True):
        self.cfg = cfg

        # create run directory
        timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        if name is not None:
            if timestamp:
                self.run_name = f"{timestamp_str}_{name}"
            else:
                self.run_name = name
        else:
            self.run_name = timestamp_str

        self.dir = os.path.join(root, self.run_name)
        os.makedirs(self.dir, exist_ok=True)

        # save config
        self.config_path = os.path.join(self.dir, "config.yaml")
        self._save_config()

        # logging
        if logger is None:
            metrics_path = os.path.join(self.dir, 'metrics.jsonl')
            self.metric_logger = JSONLLogger(metrics_path)
        else:
            self.metric_logger = logger

        events_path = os.path.join(self.dir, 'events.log')
        self.event_logger = EventLogger(events_path, console=console)

        self.event_logger.info('Spawning Manager...')
        self.event_logger.info(f'Run directory: {self.dir}')
        
    def _save_config(self):
        """ Helper function to save the experiment configuration.
        Config types supported: DataClass, OmegaConf, Dict, etc. 
        """
        with open(self.config_path, "w") as f:
            cfg = self.cfg

            # --- dataclass ---
            try:
                from dataclasses import is_dataclass, asdict
                if is_dataclass(cfg):
                    yaml.dump(asdict(cfg), f)
                    return
            except:
                pass

            # --- OmegaConf ---
            try:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(cfg):
                    yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)
                    return
            except:
                pass

            # --- dict fallback ---
            if isinstance(cfg, dict):
                yaml.dump(cfg, f)
                return

            # --- generic fallback (best effort) ---
            yaml.dump(vars(cfg), f)

    def path(self, *parts):
        """Convenience for constructing paths inside run dir"""
        return os.path.join(self.dir, *parts)
