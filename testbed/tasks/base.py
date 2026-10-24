"""Abstract interface for BED tasks."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from torch import nn

from ..core.design_spaces import DefaultDesignSpace, DesignSpace
from ..core.distributions import Distribution, as_sample_shape
from ..core.util import broadcast_and_flatten


class Task(ABC, nn.Module):
    r"""Abstract base class for Bayesian experimental design tasks.

    A :class:`Task` defines a probabilistic model consisting of

    - a prior over model parameters,
    - a likelihood (explicit or implicit),
    - methods for sampling observations, and
    - optionally, methods for evaluating likelihood log-probabilities.

    Concrete subclasses implement the task-specific prior and likelihood,
    while inheriting the common broadcasting and sampling interface.

    Args:
        event_shape_y (iterable):
            Shape of observations.
        event_shape_theta (iterable):
            Shape of parameters.
        event_shape_design (iterable):
            Shape of designs.
        prior (Distribution):
            Prior over model parameters. Its ``event_shape`` must equal
            ``event_shape_theta``, its ``batch_shape`` must be empty, and its
            ``log_prob`` must reduce over all of theta's event dims;
            ``__init__`` checks this on one prior sample.
        discrete_y (bool):
            Whether observations are discrete (default: False).
        discrete_theta (bool):
            Whether parameters are discrete (default: False).
        memoryless (bool):
            Whether outcomes don't depend on history, i.e. ``_outcome_logp``/
            ``_sample_outcome`` ignore their ``y_history``/``design_history``
            kwargs (default: True).
        implicit_likelihood (bool):
            Whether the likelihood is implicit (default: False).
        design_space (optional):
            Constraints on the space of designs. Its event shape must match
            ``event_shape_design`` (default: None).
        validate_designs (bool):
            Whether outcome_logp/sample_outcome/history_logp/sample_joint check
            incoming designs against design_space.contains(...) and raise if a
            design lies outside it. Disable for tight loops where the caller
            already guarantees valid designs (default: True).
        reparameterize_outcomes (bool, optional):
            Whether sampled outcomes carry a pathwise gradient to the design.
            None follows the subclass's :attr:`reparameterizable_outcomes`;
            True raises if the subclass cannot honour it. Set False for
            score-function estimators, which are derived with the history held
            fixed (default: None).

    Shape:
        Every tensor consists of

        - batch dimensions
        - event dimensions

        For example, a location-finding task with three sources in five
        spatial dimensions has

        - ``event_shape_theta = [3, 5]``
        - ``event_shape_design = [5]``
        - ``event_shape_y = [1]``
    """

    @property
    @abstractmethod
    def reparameterizable_outcomes(self) -> bool:
        """Whether :meth:`_sample_outcome` can sample with a pathwise gradient.

        A fact about the model, not a preference: False for genuinely discrete
        outcomes. Pathwise estimators require it; score-function estimators
        require its absence, so every task must state it explicitly.
        """
        ...

    @property
    def reparameterize_outcomes(self) -> bool:
        """Whether :meth:`sample_outcome` currently returns attached outcomes.

        Resolved against :attr:`reparameterizable_outcomes` in ``__init__``,
        which rejects a pathwise request the task cannot honour.
        """
        return self._reparameterize_outcomes

    def __init__(
        self,
        event_shape_y: Iterable[int],
        event_shape_theta: Iterable[int],
        event_shape_design: Iterable[int],
        prior: Distribution,
        discrete_y: bool = False,
        discrete_theta: bool = False,
        memoryless: bool = True,
        implicit_likelihood: bool = False,
        design_space: DesignSpace | None = None,
        validate_designs: bool = True,
        reparameterize_outcomes: bool | None = None,
    ):
        """Initializes the task."""
        nn.Module.__init__(self)

        self.event_shape_y = torch.Size(event_shape_y)
        self.event_shape_theta = torch.Size(event_shape_theta)
        self.event_shape_design = torch.Size(event_shape_design)

        self._check_prior(prior)
        self._check_prior_draw(prior)
        self.prior = prior
        self._current_prior = prior

        self.memoryless = memoryless

        self.implicit_likelihood = implicit_likelihood

        # Tracking if there are discrete components in the model
        self.discrete_y = discrete_y
        self.discrete_theta = discrete_theta

        resolved_design_space = (
            design_space
            if design_space is not None
            else DefaultDesignSpace(self.event_shape_design)
        )
        if resolved_design_space.event_shape != self.event_shape_design:
            raise ValueError(
                "design_space.event_shape must match event_shape_design; "
                f"expected {tuple(self.event_shape_design)}, got "
                f"{tuple(resolved_design_space.event_shape)}."
            )
        self.design_space = resolved_design_space
        self.validate_designs = validate_designs

        # None follows the task's own capability. An explicit True the task
        # cannot honour is an error: a
        # pathwise estimator would otherwise run on a task that never
        # supplies the gradient.
        if reparameterize_outcomes and not self.reparameterizable_outcomes:
            raise ValueError(
                f"{type(self).__name__} was constructed with "
                "reparameterize_outcomes=True, but its outcomes are not "
                "reparameterizable (see its reparameterizable_outcomes "
                "docstring for why). Pass False, or None to follow the task's "
                "own capability, and train with a score-function estimator "
                "such as REINFORCE."
            )
        if reparameterize_outcomes is None:
            reparameterize_outcomes = self.reparameterizable_outcomes
        self._reparameterize_outcomes = bool(reparameterize_outcomes)

    @property
    def device(self) -> torch.device:
        """Infers device from the first registered buffer or parameter."""
        try:
            return next(self.buffers()).device
        except StopIteration:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

    def _check_prior(self, prior):
        """Raises if prior's declared shapes break the contract for this task."""
        if prior.event_shape != self.event_shape_theta or prior.batch_shape:
            raise ValueError(
                f"{type(self).__name__}'s prior must have event_shape "
                f"{tuple(self.event_shape_theta)} and an empty batch_shape; got "
                f"event_shape {tuple(prior.event_shape)} and batch_shape "
                f"{tuple(prior.batch_shape)}. A TorchDistribution folds its "
                "batch dims into the event unless reinterpreted_batch_ndims "
                "says otherwise."
            )

    def _check_prior_draw(self, prior):
        """Raises if one prior draw breaks the shape contract for this task.

        Checks that the draw has shape ``event_shape_theta`` and that
        ``log_prob`` reduces it to a scalar. The draw uses a forked RNG, so
        constructing a task leaves seeded runs unchanged.
        """
        name = type(self).__name__
        device = prior.device
        rng_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            theta = prior.sample()
            if theta.shape != self.event_shape_theta:
                raise ValueError(
                    f"{name}'s prior.sample() must return shape "
                    f"{tuple(self.event_shape_theta)}; got {tuple(theta.shape)}."
                )
            try:
                logp = prior.log_prob(theta)
            except NotImplementedError:
                # sample-only priors have nothing further to check
                return
        if logp.shape != torch.Size():
            raise ValueError(
                f"{name}'s prior.log_prob must reduce over all event dims, "
                "mapping [*batch, *event_shape_theta] to [*batch]; for a single "
                f"draw it returned shape {tuple(logp.shape)} instead of ()."
            )

    @property
    def current_prior(self) -> Distribution:
        """The current prior over parameters, possibly updated in sequential experimental design."""
        return self._current_prior

    def null_history(self, b: int | None = None):
        """Empty (y, design) histories.

        With b=None there is no batch dim, so broadcasting can happen automatically.
        Specifying the batch dim can be useful in contexts where broadcasting is not handled.

        Args:
            b (int, optional):
                Batch size. None gives no batch dim (default: None).

        Returns:
            Tuple (Tensors):
                Empty histories on the task's device. Shapes ``[b, 0, *event_shape]`` when b is given, ``[0, *event_shape]`` otherwise.
        """
        batch = () if b is None else (b,)
        return (
            torch.empty(*batch, 0, *self.event_shape_y, device=self.device),
            torch.empty(*batch, 0, *self.event_shape_design, device=self.device),
        )

    @abstractmethod
    def _outcome_logp(self, y, theta, design, y_history, design_history):
        """Evaluates outcome log probabilities.

        The inputs (y, theta, design) are assumed to be paired,
        so that no broadcasting has to happen in this function.

        The ``y_history`` and ``design_history`` tensors are possibly
        used to condition the likelihood, in tasks which are not memoryless.
        Memoryless tasks will ignore these arguments.
        Pass in ``None`` or a tensor with ``t=0`` to indicate no history.

        Args:
            y (Tensor):
                Observations of shape ``[b, *event_shape_y]``.
            theta (Tensor):
                Parameters of shape ``[b, *event_shape_theta]``.
            design (Tensor):
                Designs of shape ``[b, *event_shape_design]``.
            y_history (Tensor):
                Past observations that the likelihood possibly conditions on. Shape ``[b, t, *event_shape_y]``.
            design_history (Tensor):
                Past designs that the likelihood possibly conditions on. Shape ``[b, t, *event_shape_design]``.

        Returns:
            Tensor:
                Log probability per batch element, shape ``[b]``.
        """
        ...

    @abstractmethod
    def _sample_outcome(self, theta, design, y_history, design_history):
        """Samples a single outcome given parameters and designs.

        The inputs (theta, design) are assumed to be paired,
        so that no broadcasting has to happen in this function.

        The ``y_history`` and ``design_history`` tensors are possibly
        used to condition the likelihood, in tasks which are not memoryless.
        Memoryless tasks will ignore these arguments.
        Pass in ``None`` or a tensor with ``t=0`` to indicate no history.

        Always returns the reparameterized draw where the model admits one, so
        implementations need not consult ``reparameterize_outcomes``. Gating
        whether those gradients are kept is :meth:`_gate_outcome_gradients`'s
        job. Sample through :meth:`sample_outcome` or :meth:`sample_joint`
        rather than calling this internal method directly, or the gate is skipped.

        Args:
            theta (Tensor):
                Parameters of shape ``[b, *event_shape_theta]``.
            design (Tensor):
                Designs of shape ``[b, *event_shape_design]``.
            y_history (Tensor):
                Past observations that the likelihood possibly conditions on. Shape ``[b, t, *event_shape_y]``.
            design_history (Tensor):
                Past designs that the likelihood possibly conditions on. Shape ``[b, t, *event_shape_design]``.

        Returns:
            Tensor:
                One sampled observation y for each (theta, design) pair, shape ``[b, *event_shape_y]``.
        """
        ...

    def _gate_outcome_gradients(self, y_flat, design_flat):
        """Detaches or validates a raw ``_sample_outcome`` draw.

        Every public sampler routes its draw through here so the
        reparameterization contract holds wherever outcomes are produced.

        Args:
            y_flat (Tensor):
                Outcomes of shape ``[b, *event_shape_y]``.
            design_flat (Tensor):
                The designs they were sampled from, shape ``[b, *event_shape_design]``.

        Returns:
            Tensor:
                The outcomes, detached when ``reparameterize_outcomes`` is False.
        """
        if not self.reparameterize_outcomes:
            return y_flat.detach()
        if (
            torch.is_grad_enabled()
            and design_flat.requires_grad
            and not y_flat.requires_grad
        ):
            # The design carried a graph but the outcome did not, so
            # _sample_outcome sampled without reparameterizing. Pathwise
            # estimators would silently lose the distributional term.
            raise ValueError(
                f"{type(self).__name__}.reparameterize_outcomes is True but "
                "_sample_outcome returned an outcome with no gradient to the "
                "design. Sample reparameterized (e.g. loc + scale * randn_like, "
                "or rsample), or declare reparameterizable_outcomes = False."
            )
        return y_flat

    def _check_design(self, design):
        """Raises if design lies outside self.design_space (when validate_designs)."""
        if not self.validate_designs:
            return
        if not self.design_space.contains(design):
            # contains() rejects both bad values and a bad event shape
            design = torch.as_tensor(design)
            raise ValueError(
                f"Design lies outside {self.design_space} "
                f"(expected trailing shape "
                f"{tuple(self.design_space.event_shape)}). "
                f"Got design of shape {tuple(design.shape)}: {design}"
            )

    def _check_history(self, y_history, design_history):
        """Normalizes histories to tensors, checking T when both are given.

        ``None`` means "not supplied" and becomes the ``T=0`` entry of
        :attr:`null_history`, so a task reading only one side can be handed
        only that side. Two supplied histories describe one trajectory, so
        their T must agree; a lone history keeps its own T.
        """
        null_y_history, null_design_history = self.null_history()

        if design_history is not None:
            self._check_design(design_history)

        if y_history is not None and design_history is not None:
            T_y = y_history.shape[-(len(self.event_shape_y) + 1)]
            T_design = design_history.shape[-(len(self.event_shape_design) + 1)]
            if T_y != T_design:
                raise ValueError(
                    f"History lengths must agree. Got y_history of shape "
                    f"{tuple(y_history.shape)} and design_history of shape "
                    f"{tuple(design_history.shape)}."
                )

        return (
            null_y_history if y_history is None else y_history,
            null_design_history if design_history is None else design_history,
        )

    def _context_history(self, y_history, design_history):
        """Parses a history that is used as conditioning context.

        If a task is memoryless, return the null history so that we can
        avoid paying to broadcast it. Otherwise, just check the history
        is valid.

        Not for :meth:`history_logp`, where the history is the
        data being scored rather than conditioning variables.
        """
        if self.memoryless:
            return self.null_history()
        return self._check_history(y_history, design_history)

    def outcome_logp(self, y, theta, design, y_history=None, design_history=None):
        """Evaluates outcome log probabilities with broadcasting.

        Broadcasting conventions follow ``torch.distributions``.

        Args:
            y (Tensor):
                Observations of shape ``[*batch_shape_y, *event_shape_y]``.
            theta (Tensor):
                Parameters of shape ``[*batch_shape_theta, *event_shape_theta]``.
            design (Tensor):
                Designs of shape ``[*batch_shape_design, *event_shape_design]``.
            y_history (Tensor, optional):
                Past observations that the likelihood possibly conditions on. Shape ``[**batch_shape_history, t, *event_shape_y]``.
            design_history (Tensor, optional):
                Past designs that the likelihood possibly conditions on. Shape ``[**batch_shape_history, t, *event_shape_design]``.

        Returns:
            Tensor:
                Log probabilities, shape ``[*broadcast_batch_shape]``.
        """
        self._check_design(design)
        y_history, design_history = self._context_history(y_history, design_history)

        # T_y and T_design agree unless only one history was supplied.
        T_y = y_history.shape[-(len(self.event_shape_y) + 1)]
        T_design = design_history.shape[-(len(self.event_shape_design) + 1)]
        (
            (y_flat, theta_flat, design_flat, y_history_flat, design_history_flat),
            batch_shape,
        ) = broadcast_and_flatten(
            (y, self.event_shape_y),
            (theta, self.event_shape_theta),
            (design, self.event_shape_design),
            (y_history, (T_y, *self.event_shape_y)),
            (design_history, (T_design, *self.event_shape_design)),
        )
        logp_flat = self._outcome_logp(
            y_flat,
            theta_flat,
            design_flat,
            y_history=y_history_flat,
            design_history=design_history_flat,
        )
        return logp_flat.reshape(batch_shape)

    def _compiled(self, name):
        """Returns the method ``name`` compiled, compiling on first use.

        Cached on the task: a fresh ``torch.compile`` wrapper per call would
        recompile every time.
        """
        key = f"_compiled_{name}"
        fn = self.__dict__.get(key)
        if fn is None:
            fn = torch.compile(getattr(self, name), dynamic=False)
            self.__dict__[key] = fn
        return fn

    def history_logp(
        self,
        y_history,
        design_history,
        theta,
        y_context=None,
        design_context=None,
        fused=False,
    ):
        r"""Computes log probabilities for a sequence of observations and designs.

        Sums the per-step log-prob, each conditioned on the history strictly before it:

        .. math::
            p(y_{1:T}, x_{1:T} \mid \theta) = \prod_{t=1}^T p(y_t \mid \theta, x_t, y_{<t}, x_{<t})

        Given a context :math:`(y_{1:S}, x_{1:S})`, scores the trajectory that
        *follows* it, conditioning on the context at every step but never
        scoring it:

        .. math::
            p(y_{1:T}, x_{1:T} \mid \theta, y_{1:S}, x_{1:S})
            = \prod_{t=1}^T p(y_t \mid \theta, x_t, y_{1:S}, x_{1:S}, y_{<t}, x_{<t})

        Use a context when the information in it is already carried elsewhere --
        e.g. by :attr:`current_prior` after a posterior update -- so that
        scoring it again would count it twice.

        Correct for memoryless and history-dependent likelihoods alike, since
        ``_outcome_logp`` receives ``y_history``/``design_history`` at every
        step and a memoryless task simply ignores them. Override only if the
        likelihood isn't expressible as such a per-step product at all.

        Args:
            y_history (Tensor):
                Observations of shape ``[*batch_shape_y, T, *event_shape_y]``.
            design_history (Tensor):
                Designs of shape ``[*batch_shape_design, T, *event_shape_design]``.
            theta (Tensor):
                Parameters of shape ``[*batch_shape_theta, *event_shape_theta]``.
            y_context (Tensor, optional):
                Earlier observations to condition on without scoring. Shape
                ``[*batch_shape_context, S, *event_shape_y]`` (default: None).
            design_context (Tensor, optional):
                The designs alongside them, shape
                ``[*batch_shape_context, S, *event_shape_design]`` (default: None).
            fused (bool):
                Score the steps with compiled kernels instead of the eager loop.
                A memoryless task folds T into the batch via
                :meth:`_memoryless_history_logp`, ignoring the context.
                A history-dependent task compiles the T loop itself via :meth:`_sequential_history_logp` (default: False).

        Returns:
            Tensor:
                Log probabilities, shape ``[*broadcast_batch_shape]``.
        """
        y_history, design_history = self._check_history(y_history, design_history)

        if not fused:
            # both memoryless and non-memoryless tasks loop over T when not fused
            # speedups for using _memoryless_history_logp are small but cost
            # T times as much memory
            return self._sequential_history_logp(
                y_history, design_history, theta, y_context, design_context
            )
        if self.memoryless:
            return self._compiled("_memoryless_history_logp")(
                y_history, design_history, theta
            )
        return self._compiled("_sequential_history_logp")(
            y_history, design_history, theta, y_context, design_context
        )

    def _memoryless_history_logp(self, y_history, design_history, theta):
        r"""Computes logprob for a rollout with T folded into the batch.

        For efficient calculation on memoryless tasks: one ``_outcome_logp``
        call over ``[*batch_shape * T, ...]`` replaces the per-step loop.

        Reached through ``history_logp(..., fused=True)``, which compiles it.
        Only worth calling compiled. Without compilation, materializes all
        ``[b_flat * T, ...]`` entries at once, yielding negligible speedups
        for large memory costs.
        """
        T = y_history.shape[-(len(self.event_shape_y) + 1)]
        batch_shape = torch.broadcast_shapes(
            y_history.shape[: y_history.ndim - 1 - len(self.event_shape_y)],
            design_history.shape[
                : design_history.ndim - 1 - len(self.event_shape_design)
            ],
            theta.shape[: theta.ndim - len(self.event_shape_theta)],
        )
        step_shape = (*batch_shape, T)

        def flat(x, event_shape):
            """Broadcasts x, which already carries a T axis, over the steps."""
            return x.expand(*step_shape, *event_shape).reshape(-1, *event_shape)

        # theta is shared across steps; give a T axis to broadcast along
        theta = theta.unsqueeze(theta.ndim - len(self.event_shape_theta))
        logp = self._outcome_logp(
            flat(y_history, self.event_shape_y),
            flat(theta, self.event_shape_theta),
            flat(design_history, self.event_shape_design),
            y_history=None,
            design_history=None,
        )
        return logp.reshape(step_shape).sum(-1)

    def _sequential_history_logp(
        self, y_history, design_history, theta, y_context, design_context
    ):
        """Scores the steps one at a time, each conditioned on its full prefix."""
        # Context becomes null for memoryless tasks
        y_context, design_context = self._context_history(y_context, design_context)

        # T must agree for y_history and design_history as they are paired
        # S can vary across the context (e.g. only one could be supplied)
        T = y_history.shape[-(len(self.event_shape_y) + 1)]
        S_y = y_context.shape[-(len(self.event_shape_y) + 1)]
        S_design = design_context.shape[-(len(self.event_shape_design) + 1)]
        (
            (y_flat, design_flat, theta_flat, y_context_flat, design_context_flat),
            batch_shape,
        ) = broadcast_and_flatten(
            (y_history, (T, *self.event_shape_y)),
            (design_history, (T, *self.event_shape_design)),
            (theta, tuple(self.event_shape_theta)),
            (y_context, (S_y, *self.event_shape_y)),
            (design_context, (S_design, *self.event_shape_design)),
        )

        # skips a copy op when context is null
        y_full = y_flat if S_y == 0 else torch.cat([y_context_flat, y_flat], dim=1)
        design_full = (
            design_flat
            if S_design == 0
            else torch.cat([design_context_flat, design_flat], dim=1)
        )

        b_flat = theta_flat.shape[0]
        logp = torch.zeros(b_flat, device=theta_flat.device)
        for t in range(T):
            logp += self._outcome_logp(
                y_flat[:, t],
                theta_flat,
                design_flat[:, t],
                y_history=y_full[:, : S_y + t],
                design_history=design_full[:, : S_design + t],
            )
        return logp.reshape(batch_shape)

    @staticmethod
    def _expand_samples(n, *tensors):
        """Repeats each ``[b, *rest]`` tensor n times into ``[n * b, *rest]``.

        Sample-major, so that reshaping to ``[n, b, ...]`` recovers the sample dim.
        """
        return tuple(x.expand(n, *x.shape).flatten(0, 1) for x in tensors)

    def sample_outcome(
        self, theta, design, y_history=None, design_history=None, sample_shape=()
    ):
        """Samples outcomes from the likelihood.

        Detaches the sample when ``reparameterize_outcomes`` is False, so the
        outcome is graph-free before it reaches a designer. Detaching here
        rather than downstream also stops later designs acquiring a gradient
        path through it via the policy's history. Always sample through this
        wrapper, never via ``_sample_outcome`` directly.

        Args:
            theta (Tensor):
                Parameters of shape ``[*batch_shape_theta, *event_shape_theta]``.
            design (Tensor):
                Designs of shape ``[*batch_shape_design, *event_shape_design]``.
            y_history (Tensor, optional):
                Past observations that the likelihood possibly conditions on. Shape ``[**batch_shape_history, t, *event_shape_y]``.
            design_history (Tensor, optional):
                Past designs that the likelihood possibly conditions on. Shape ``[**batch_shape_history, t, *event_shape_design]``.
            sample_shape (int or iterable):
                Shape of the observation samples drawn for each (theta, design)
                pair, as in ``torch.distributions``. An int ``n`` means ``(n,)``
                (default: ()).

        Returns:
            Tensor:
                Sampled observations, shape ``[*sample_shape, *broadcast_batch_shape, *event_shape_y]``.
        """
        sample_shape = as_sample_shape(sample_shape)
        self._check_design(design)
        y_history, design_history = self._context_history(y_history, design_history)

        # T_y and T_design agree unless only one history was supplied.
        T_y = y_history.shape[-(len(self.event_shape_y) + 1)]
        T_design = design_history.shape[-(len(self.event_shape_design) + 1)]
        (theta_flat, design_flat, y_history_flat, design_history_flat), batch_shape = (
            broadcast_and_flatten(
                (theta, self.event_shape_theta),
                (design, self.event_shape_design),
                (y_history, (T_y, *self.event_shape_y)),
                (design_history, (T_design, *self.event_shape_design)),
            )
        )
        theta_flat, design_flat, y_history_flat, design_history_flat = (
            self._expand_samples(
                sample_shape.numel(),
                theta_flat,
                design_flat,
                y_history_flat,
                design_history_flat,
            )
        )
        y_flat = self._gate_outcome_gradients(
            self._sample_outcome(
                theta_flat,
                design_flat,
                y_history=y_history_flat,
                design_history=design_history_flat,
            ),
            design_flat,
        )
        return y_flat.reshape(*sample_shape, *batch_shape, *self.event_shape_y)

    def sample_joint(
        self, design, y_history=None, design_history=None, sample_shape=()
    ):
        r"""Samples from the joint over (y, theta) given designs.

        Uses the current (possibly updated) prior. Gates the outcome's
        gradients exactly as :meth:`sample_outcome` does, so the outcome is
        graph-free when ``reparameterize_outcomes`` is False.

        Args:
            design (Tensor):
                Designs of shape ``[*batch_shape_designs, *event_shape_design]``.
            y_history (Tensor, optional):
                Past observations that the likelihood possibly conditions on. Shape ``[**batch_shape_history, t, *event_shape_y]``.
            design_history (Tensor, optional):
                Past designs that the likelihood possibly conditions on. Shape ``[**batch_shape_history, t, *event_shape_design]``.
            sample_shape (int or iterable):
                Shape of the samples drawn for each design. An int ``n``
                means ``(n,)`` (default: ()).

        Returns:
            Tensor:
                Sampled observations, shape ``[*sample_shape, *broadcast_batch_shape, *event_shape_y]``.
            Tensor:
                Sampled parameters, shape ``[*sample_shape, *broadcast_batch_shape, *event_shape_theta]``.
        """
        sample_shape = as_sample_shape(sample_shape)
        self._check_design(design)
        y_history, design_history = self._context_history(y_history, design_history)

        # T_y and T_design agree unless only one history was supplied.
        T_y = y_history.shape[-(len(self.event_shape_y) + 1)]
        T_design = design_history.shape[-(len(self.event_shape_design) + 1)]
        (
            (design_flat, y_history_flat, design_history_flat),
            batch_shape,
        ) = broadcast_and_flatten(
            (design, self.event_shape_design),
            (y_history, (T_y, *self.event_shape_y)),
            (design_history, (T_design, *self.event_shape_design)),
        )
        design_flat, y_history_flat, design_history_flat = self._expand_samples(
            sample_shape.numel(), design_flat, y_history_flat, design_history_flat
        )
        total = design_flat.shape[0]
        theta_flat = self.sample_current_prior(total)  # [total, *event_theta]
        y_flat = self._gate_outcome_gradients(
            self._sample_outcome(
                theta_flat,
                design_flat,
                y_history=y_history_flat,
                design_history=design_history_flat,
            ),
            design_flat,
        )  # [total, *event_y]
        return (
            y_flat.reshape(*sample_shape, *batch_shape, *self.event_shape_y),
            theta_flat.reshape(*sample_shape, *batch_shape, *self.event_shape_theta),
        )

    def sample_marginal(
        self, design, y_history=None, design_history=None, sample_shape=()
    ):
        r"""Samples from the marginal over observations given designs.

        Uses the current (possibly updated) prior.

        Args:
            design (Tensor):
                Designs of shape ``[*batch_shape_designs, *event_shape_design]``.
            y_history (Tensor, optional):
                Past observations that the likelihood possibly conditions on. Shape ``[**batch_shape_history, t, *event_shape_y]``.
            design_history (Tensor, optional):
                Past designs that the likelihood possibly conditions on. Shape ``[**batch_shape_history, t, *event_shape_design]``.
            sample_shape (int or iterable):
                Shape of the samples drawn for each design. An int ``n``
                means ``(n,)`` (default: ()).

        Returns:
            Tensor:
                Sampled observations, shape ``[*sample_shape, *broadcast_batch_shape, *event_shape_y]``.
        """
        outcome, _ = self.sample_joint(
            design, y_history, design_history, sample_shape=sample_shape
        )
        return outcome

    def sample_prior(self, sample_shape=()):
        r"""Samples from original, non-updated prior.

        Args:
            sample_shape (int or iterable):
                Shape of the batch of draws. An int ``n`` means ``(n,)``
                (default: ()).

        Returns:
            Tensor:
                Sampled parameters, shape ``[*sample_shape, *event_shape_theta]``.
        """
        return self.prior.sample(sample_shape)

    def sample_current_prior(self, sample_shape=()):
        r"""Samples from current, possibly updated prior.

        Args:
            sample_shape (int or iterable):
                Shape of the batch of draws. An int ``n`` means ``(n,)``
                (default: ()).

        Returns:
            Tensor:
                Sampled parameters, shape ``[*sample_shape, *event_shape_theta]``.
        """
        return self.current_prior.sample(sample_shape)

    def update_prior(self, new_prior):
        r"""Updates the current prior.

        Args:
            new_prior (Distribution):
                Prior to update to. Must have the same ``event_shape`` as the
                original prior and an empty ``batch_shape``, and its
                ``sample`` and ``log_prob`` must honour them; this is checked
                on one draw, as in ``__init__``.
        """
        self._check_prior(new_prior)
        self._check_prior_draw(new_prior)
        self._current_prior = new_prior
