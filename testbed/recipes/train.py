import torch

from testbed.designers import PolicyDesigner

"""
A simple end-to-end training recipe for policy networks.
"""

class Trainer:
    def __init__(
        self,
        task,
        designer,
        objective,
        optimizer,
        cfg,
        manager=None
    ):
        assert isinstance(designer, PolicyDesigner), 'This Trainer assumes your designer is a PolicyDesigner object.'

        self.task = task
        self.designer = designer
        self.objective = objective
        self.optimizer = optimizer
        self.cfg = cfg
        self.manager = manager

        # checkpointing
        self.best_metric = -torch.inf

    @property
    def metric_logger(self):
        if self.manager is None:
            return None
        return self.manager.metric_logger

    def train(self):
        self.designer.policy_net.train()

        for step in range(self.cfg.n_train_iter):
            metrics = self.train_step()
            metrics['step'] = step

            if self.metric_logger is not None:
                self.metric_logger.log(metrics)

            # evaluation
            if self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                eval_metrics['step'] = step

                if self.metric_logger is not None:
                    self.metric_logger.log(eval_metrics)

            self._maybe_save_best(step, metrics, min_step=500)

        if self.manager is not None:
            save_path = self.manager.path("final_step.pt")
            torch.save({
                "model": self.designer.policy_net.state_dict(),
                "step": step,
            }, save_path)
    
        if self.metric_logger is not None:
            self.metric_logger.close()

    def train_step(self):
        cfg = self.cfg

        theta0 = self.task.sample_prior(cfg.n_outer)
        y_obs, xi_obs = self.designer.rollout(theta0)

        gain = self.objective.estimate_gain(y_obs, xi_obs, theta0)
        loss = -gain

        self.optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.designer.policy_net.parameters(), cfg.grad_clip
            )

        self.optimizer.step()

        return {
            'gain': gain.item(),
        }

    def evaluate(self):
        # TODO
        # Evaluating with a large inner sample size slows down training quite a bit. 
        # It's also memory intensive and kills my local machine. 

        return {
            'eval/spce': 0.0,
            'eval/snmc': 0.0
        }

    def _maybe_save_best(self, step, metrics, min_step=0):
        # Saves a checkpoint of the model if it's the lowest training loss seen so far
        # NB: assumes our objective is maximizing a gain
        
        # min_step controls how long we wait before saving; useful for avoiding a ton of i/o at the start
        if step < min_step:
            return

        if self.manager is None:
            return

        step = metrics['step']
        metric_val = metrics['gain']
        is_better = metric_val > self.best_metric

        if is_better:
            self.best_metric = metric_val

            chkpt_path = self.manager.path("best.pt")

            torch.save({
                "model": self.designer.policy_net.state_dict(),
                "step": step,
                "metric": metric_val,
            }, chkpt_path)

            self.manager.event_logger.info(f'Saving new best checkpoint : gain={metric_val} @ step={step}')
