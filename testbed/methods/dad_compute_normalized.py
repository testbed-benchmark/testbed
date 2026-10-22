"""One-off DAD inner-sample sweep with a fixed wall-clock training budget.

Uses the ordinary DAD update and network/objective factories. Setup and final
serialization/evaluation are outside the budget; updates, synchronization,
in-memory policy snapshots, and metric logging are inside it. Snapshots ensure
an unexpectedly slow update cannot put post-deadline weights in final_step.pt.
"""

import math
import time
from collections import deque

import hydra
import torch

from testbed.core.setup import setup
from testbed.designers import PolicyDesigner
from testbed.methods.dad import Trainer as DADTrainer
from testbed.methods.dad import build_design_net, build_objective


class Trainer(DADTrainer):
    """Train DAD until the next update is unlikely to fit the remaining time."""

    def _synchronize(self):
        device = torch.device(self.cfg.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def _snapshot(self):
        # copy=True matters on CPU too: state_dict tensors alias live weights.
        return {
            key: value.detach().to(device="cpu", copy=True)
            for key, value in self.designer.design_net.state_dict().items()
        }

    def train(self):
        """Save the last policy whose synchronized update finished in budget.

        A rolling maximum of the last 20 iterations, times a safety factor,
        predicts whether another iteration fits. An update cannot be interrupted
        safely: if it overruns anyway, save the preceding snapshot. Consequently
        process runtime can exceed the budget, but saved weights never include
        an update that completed after it. No best-loss checkpoint is selected.
        """
        budget = float(self.cfg.train_time_budget_seconds)
        safety_factor = float(self.cfg.step_time_safety_factor)
        max_steps = self.cfg.n_train_iter
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("train_time_budget_seconds must be finite and positive")
        if not math.isfinite(safety_factor) or safety_factor < 1:
            raise ValueError("step_time_safety_factor must be finite and >= 1")
        if max_steps is not None and (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < 1
        ):
            raise ValueError("n_train_iter must be null or a positive integer")
        if self.cfg.eval_every != 0:
            raise ValueError("Use eval_every=0; evaluate final_step.pt after training")
        if self.manager is None:
            raise ValueError("A run manager is required to save the timed checkpoint")

        self.designer.design_net.train()
        self._synchronize()
        state = self._snapshot()
        completed_steps = 0
        attempted_steps = 0
        checkpoint_elapsed = 0.0
        durations = deque(maxlen=20)
        logger = self.manager.metric_logger
        start = time.monotonic()
        stop_reason = "iteration_limit"

        while max_steps is None or completed_steps < max_steps:
            iteration_start = time.monotonic()
            remaining = budget - (iteration_start - start)
            if remaining <= 0:
                stop_reason = "time_budget"
                break
            if durations and remaining < max(durations) * safety_factor:
                stop_reason = "insufficient_time_for_next_step"
                break

            metrics = self.train_step()
            self._synchronize()
            attempted_steps += 1
            elapsed = time.monotonic() - start
            if elapsed > budget:
                stop_reason = "step_overran_budget"
                break

            state = self._snapshot()
            checkpoint_elapsed = elapsed
            completed_steps += 1
            metrics.update(
                step=completed_steps - 1,
                **{
                    "train/elapsed_seconds": elapsed,
                    "train/completed_steps": completed_steps,
                },
            )
            if logger is not None:
                logger.log(metrics)
            durations.append(time.monotonic() - iteration_start)

        elapsed = time.monotonic() - start
        checkpoint = {
            "design_net": state,
            "use_raw_history": self.designer.use_raw_history,
            "step": completed_steps - 1,
            "completed_steps": completed_steps,
            "attempted_steps": attempted_steps,
            "train_time_budget_seconds": budget,
            "checkpoint_elapsed_seconds": checkpoint_elapsed,
            "training_elapsed_seconds": elapsed,
            "stop_reason": stop_reason,
        }
        torch.save(checkpoint, self.manager.path("final_step.pt"))
        if completed_steps == 0:
            self.manager.event_logger.warning(
                "No update finished within the budget; saved initial policy (step=-1)."
            )
        self.manager.event_logger.info(
            f"Saved final_step.pt after {completed_steps} updates "
            f"({checkpoint_elapsed:.3f}s / {budget:.3f}s); {stop_reason}. "
            f"Training loop took {elapsed:.3f}s."
        )
        if logger is not None:
            logger.log(
                {
                    "step": max(completed_steps - 1, 0),
                    "train/completed_steps": completed_steps,
                    "train/attempted_steps": attempted_steps,
                    "train/time_budget_seconds": budget,
                    "train/checkpoint_elapsed_seconds": checkpoint_elapsed,
                    "train/elapsed_seconds": elapsed,
                    "train/stop_reason": stop_reason,
                }
            )
            logger.close()


@hydra.main(
    version_base="1.3",
    config_path="../recipes",
    config_name="experiments/location_finding_2src/dad_compute_normalized",
)
def main(cfg):
    """Build the standard DAD components and run the timed training loop."""
    manager, task = setup(cfg)
    torch.manual_seed(cfg.seed)
    policy_net = build_design_net(cfg, task).to(cfg.device)
    designer = PolicyDesigner(
        task,
        cfg.T,
        policy_net,
        reparameterize_policy=cfg.objective.reparameterize_policy,
        use_raw_history=cfg.use_raw_history,
    )
    cfg.objective.requires_policy_log_probs = designer.requires_policy_log_probs
    objective = build_objective(cfg, task)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg.lr, betas=cfg.betas)
    scheduler = None
    if cfg.lr_anneal_every is not None and cfg.lr_gamma is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_anneal_every, gamma=cfg.lr_gamma
        )
    Trainer(task, designer, objective, optimizer, cfg, manager, scheduler).train()


if __name__ == "__main__":
    main()
