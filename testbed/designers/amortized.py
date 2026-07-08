"""Amortized designer abstractions."""

from abc import ABC, abstractmethod

import torch

from ..design_nets.stochastic import StochasticDesignNet


class Designer(ABC):
    """Base class for rollout designers."""

    def __init__(self, task, T):
        """Stores the task and horizon."""
        super().__init__()

        self.task = task
        self.T = T

    @abstractmethod
    def propose_design(self, y_history, design_history, *, mask=None):
        """Proposes the next design."""
        # y_history, design_history: [b=1, T, **dim]
        # mask: optional boolean mask [b=1, T] for padded histories
        # returns: [b=1, **dim_design]
        ...

    def _step_design(self, y_history, design_history):
        """Proposes the next design, returning (design, policy_log_prob or None)."""
        return self.propose_design(y_history, design_history), None

    def empty_history(self, b):
        """Returns an empty rollout (i.e., corresponding to t=0)."""
        y_history = torch.empty(
            (b, 0, *self.task.event_shape_y), device=self.task.device
        )
        design_history = torch.randn(
            (b, 0, *self.task.event_shape_design), device=self.task.device
        )
        return y_history, design_history

    def rollout(self, theta, return_policy_log_probs: bool = False):
        """Runs a rollout for batched theta values."""
        # Runs the designer given a true theta for T steps
        # theta: [b, dim_theta]
        # returns: y_history, design_history: [b, T, **]
        # with return_policy_log_probs=True, also per-step policy log probs: [b, T]
        assert theta.device == self.device, (
            f"theta is on {theta.device} but designer is on {self.device}. "
            f"Move theta with theta.to(designer.device), or move the designer with designer.to(theta.device)."
        )

        b = theta.shape[0]
        T = self.T

        y_list = []
        design_list = []
        policy_log_prob_list = []

        for _ in range(T):
            y_hist_t = (
                torch.stack(y_list, dim=1)
                if y_list
                else torch.empty(b, 0, *self.task.event_shape_y, device=theta.device)
            )
            d_hist_t = (
                torch.stack(design_list, dim=1)
                if design_list
                else torch.empty(
                    b, 0, *self.task.event_shape_design, device=theta.device
                )
            )

            design, policy_log_prob = self._step_design(y_hist_t, d_hist_t)
            y = self.task.sample_outcome(theta, design)  # [b, *event_shape_y]

            y_list.append(y)
            design_list.append(design)
            policy_log_prob_list.append(policy_log_prob)

        y_history = torch.stack(y_list, dim=1)  # [b, T, *event_shape_y]
        design_history = torch.stack(design_list, dim=1)  # [b, T, *event_shape_design]

        if not return_policy_log_probs:
            return y_history, design_history
        if any(log_prob is None for log_prob in policy_log_prob_list):
            raise RuntimeError(
                "policy log probs require a stochastic designer (deterministic=False)"
            )
        policy_log_probs = torch.stack(policy_log_prob_list, dim=1)  # [b, T]
        return y_history, design_history, policy_log_probs


class PolicyDesigner(Designer):
    """Designer backed by a policy network."""

    def __init__(
        self,
        task,
        T,
        policy_net,
        *,
        deterministic: bool | None = None,
        reparameterize_policy: bool = False,
    ):
        """Initializes with a policy network.

        deterministic: None infers from the policy type; stochastic nets roll out
        stochastically by default, so pass deterministic=True for greedy eval.
        reparameterize_policy: sample designs with the reparameterization trick
        so they carry the graph (pathwise gradients); False gives pure score log probs.
        """
        super().__init__(task, T)

        self.policy_net = policy_net
        if deterministic is None:
            deterministic = not isinstance(policy_net, StochasticDesignNet)
        self.deterministic = bool(deterministic)
        self.reparameterize_policy = bool(reparameterize_policy)

    @property
    def device(self) -> torch.device:
        """Returns the policy network device."""
        return self.policy_net.device

    @property
    def reinforce_requires_policy_log_probs(self) -> bool:
        """True when estimate_gain needs per-step policy log probs.

        Score-function form: the designer samples designs without
        reparameterization (and not greedily), so the gradient must come from
        the policy score rather than a pathwise graph.
        """
        return not self.deterministic and not self.reparameterize_policy

    def propose_design(self, y_history, design_history, *, mask=None):
        """Evaluates the policy network."""
        # y_history, design_history: [b, T, **dim]
        # mask: optional boolean mask [b, T] for padded histories
        # returns: [b, **dim_design]

        # TODO enforce policy_net can handle this kind of input or otherwise handle it here
        # When t=0, the slices of y_history, design_history will have shape [b, 0, dim]
        if self.deterministic:
            # mask is not forwarded here: policy_net.__call__ (forward) does not
            # accept a mask argument. For padded-history eval with deterministic=True,
            # call policy_net.sample_from_observation(obs, deterministic=True) directly.
            design = self.policy_net(y_history, design_history)
        else:
            design = self.policy_net.sample(y_history, design_history, mask=mask).design
        return design

    def _step_design(self, y_history, design_history):
        """Proposes the next design; stochastic policies also return the log prob."""
        if self.deterministic:
            return self.policy_net(y_history, design_history), None
        sample = self.policy_net.sample(
            y_history, design_history, reparameterize_policy=self.reparameterize_policy
        )
        return sample.design, sample.log_prob
