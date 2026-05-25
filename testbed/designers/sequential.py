import torch

from testbed.inference.is_sampler import run_importance_sampling


class SequentialDesigner:

    def __init__(self, task, T, objective, lr, n_step_opt,
                 inference_fn=run_importance_sampling,
                 init_design_dist=None):
        """
        init_design_dist controls how the initial design is drawn before each gradient optimization:
          None             -- standard normal N(0, I)  [default]
          "current_prior"  -- sample one design from task.current_prior (useful when the design
                              space and theta space coincide, e.g. source-finding)
          Distribution     -- any testbed Distribution; one sample is drawn per call
        """
        self.task = task
        self.T = T
        self.objective = objective
        self.inference_fn = inference_fn
        self.lr = lr
        self.n_step_opt = n_step_opt
        self.init_design_dist = init_design_dist

    def initialize_design(self):
        # Returns [*event_shape_design] with requires_grad=True.
        # Override this method for custom initialization logic.
        dist = self.init_design_dist
        if dist is None:
            design = torch.randn(self.task.event_shape_design, device=self.task.device)
        elif dist == "current_prior":
            design = self.task.sample_current_prior().to(self.task.device)
        else:
            design = dist.sample(1).reshape(self.task.event_shape_design).to(self.task.device)
        return design.requires_grad_(True)

    def propose_design(self, return_gain=False):
        # Selects the next design via gradient descent on the objective
        # returns: [*event_shape_design]
        design = self.initialize_design()
        optimizer = torch.optim.Adam([design], lr=self.lr)

        for _ in range(self.n_step_opt):
            optimizer.zero_grad()
            gain = self.objective(design)
            loss = -gain
            loss.backward()
            optimizer.step()

        design = design.detach()
        if return_gain:
            return design, gain.item()
        return design

    def rollout(self, theta_true):
        """
        Runs the full sequential experimental design procedure:
        Propose -> Sample -> Update posterior

        Args:
            theta_true: [*event_shape_theta] or [1, *event_shape_theta]
                        Leading size-1 batch dims are tolerated and stripped.

        Returns:
            y_history:      [T, *event_shape_y]
            design_history: [T, *event_shape_design]
        """
        n_event_dims = len(self.task.event_shape_theta)
        n_batch_dims = theta_true.ndim - n_event_dims
        batch_shape = theta_true.shape[:n_batch_dims]
        if any(s != 1 for s in batch_shape):
            raise ValueError(
                f"rollout expects a single theta_true; got shape {tuple(theta_true.shape)}"
            )
        theta_true = theta_true.reshape(*self.task.event_shape_theta)

        y_list, design_list = [], []

        for _ in range(self.T):
            design = self.propose_design()
            y = self.task.sample_outcome(theta_true, design)

            y_list.append(y)
            design_list.append(design)

            y_history = torch.stack(y_list)           # [t+1, *event_shape_y]
            design_history = torch.stack(design_list) # [t+1, *event_shape_design]

            posterior = self.inference_fn(self.task, y_history, design_history)
            self.task.update_prior(posterior)

        return torch.stack(y_list), torch.stack(design_list)
