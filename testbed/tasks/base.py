"""Abstract interface for BED tasks."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.transforms import Transform

from ..core.constraints import default_design_transform, in_support
from ..core.distributions import (
    Distribution,
    EmpiricalDistribution,
    as_distribution,
    as_sample_shape,
)
from ..core.util import broadcast_and_flatten, matches_event_shape


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
        memoryless (bool):
            Whether outcomes don't depend on history, i.e. ``_outcome_logp``/
            ``_sample_outcome`` ignore their ``y_history``/``design_history``
            kwargs (default: True).
        implicit_likelihood (bool):
            Whether the likelihood is implicit (default: False).
        design_prior (Distribution, optional):
            Distribution of random designs, whose ``support`` is the space of
            valid designs and which determines :attr:`design_transform`. Its
            ``event_shape`` must equal ``event_shape_design``. None gives a
            standard normal over unconstrained designs (default: None).
        validate_designs (bool):
            Whether outcome_logp/sample_outcome/history_logp/sample_joint check
            incoming designs against :meth:`contains_design` and raise if a
            design lies outside the design space. Disable for tight loops where
            the caller already guarantees valid designs (default: True).
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

    # history_logp scores a memoryless task's steps in one call, rather than
    # looping over T, when there are at most this many (batch size times T).
    # Much faster for small batch sizes; at the cost of additional memory.
    max_folded_steps = 2**17  # ~approx 131k

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
        memoryless: bool = True,
        implicit_likelihood: bool = False,
        design_prior: Distribution | None = None,
        validate_designs: bool = True,
        reparameterize_outcomes: bool | None = None,
    ):
        """Initializes the task."""
        nn.Module.__init__(self)

        self.event_shape_y = torch.Size(event_shape_y)
        self.event_shape_theta = torch.Size(event_shape_theta)
        self.event_shape_design = torch.Size(event_shape_design)

        prior = as_distribution(prior)
        self._validate_prior(prior)
        self.prior = prior
        self._current_prior = prior

        self.memoryless = memoryless

        self.implicit_likelihood = implicit_likelihood

        self.discrete_y = discrete_y

        if design_prior is None:
            zeros = torch.zeros(self.event_shape_design)
            design_prior = Normal(zeros, 1.0)
        design_prior = as_distribution(design_prior)
        if design_prior.event_shape != self.event_shape_design:
            raise ValueError(
                "design_prior.event_shape must match event_shape_design; "
                f"expected {tuple(self.event_shape_design)}, got "
                f"{tuple(design_prior.event_shape)}."
            )
        if isinstance(design_prior, EmpiricalDistribution):
            # designs are addressed by index, so each must appear exactly once
            locs = design_prior.locs
            if locs.numel() == 0 or torch.unique(
                locs.reshape(len(locs), -1), dim=0
            ).shape[0] != len(locs):
                raise ValueError(
                    "A finite design space needs at least one design, each listed once."
                )
        self.design_prior = design_prior
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

    def _validate_prior(self, prior):
        """Checks declared shapes and one draw without advancing the RNG."""
        if prior.event_shape != self.event_shape_theta:
            raise ValueError(
                f"{type(self).__name__}'s prior must have event_shape "
                f"{tuple(self.event_shape_theta)}; got {tuple(prior.event_shape)}. "
                "A torch distribution's batch dims are folded into its event."
            )

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
    def discrete_theta(self) -> bool:
        """Whether parameters are discrete, i.e. the prior's support is."""
        return self.prior.support.is_discrete

    @property
    def design_transform(self) -> Transform:
        """Maps raw designs onto the design space, ``design_prior.support``.

        The design prior's own ``transform`` if it defines one, and otherwise
        :func:`~testbed.core.constraints.default_design_transform` of its
        support. Cached, since designers apply it on every step; the cache is
        dropped when the task moves (``.to()``, ``.double()``, ...) or the
        design prior is replaced.
        """
        transform = self.__dict__.get("_design_transform")
        if transform is None:
            transform = getattr(self.design_prior, "transform", None)
            if transform is None:
                transform = default_design_transform(self.design_prior.support)
            self.__dict__["_design_transform"] = transform
        return transform

    def _apply(self, fn, recurse=True):
        """Drops the cached design transform, whose tensors are on the old device."""
        self.__dict__.pop("_design_transform", None)
        return super()._apply(fn, recurse)

    def __setattr__(self, name, value):
        """Drops the cached design transform when the design prior is replaced."""
        if name == "design_prior":
            self.__dict__.pop("_design_transform", None)
        super().__setattr__(name, value)

    def contains_design(self, design) -> bool:
        """Whether every design in a batch has the design shape and lies in the design space."""
        design = torch.as_tensor(design)
        return matches_event_shape(design, self.event_shape_design) and in_support(
            self.design_prior.support, design
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
        """Raises if design lies outside the design space (when validate_designs)."""
        if not self.validate_designs:
            return
        if not self.contains_design(design):
            # contains_design() rejects both bad values and a bad event shape
            design = torch.as_tensor(design)
            raise ValueError(
                f"Design lies outside the design space {self.design_prior.support} "
                f"(expected trailing shape "
                f"{tuple(self.event_shape_design)}). "
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
                Score the steps with compiled kernels instead of eagerly.
                A memoryless task folds T into the batch via
                :meth:`_memoryless_history_logp`, ignoring the context.
                A history-dependent task compiles the T loop itself via
                :meth:`_sequential_history_logp`. Eagerly, a memoryless task
                also folds T into the batch when there are at most
                :attr:`max_folded_steps` steps in total, and otherwise loops
                over T (default: False).

        Returns:
            Tensor:
                Log probabilities, shape ``[*broadcast_batch_shape]``.
        """
        y_history, design_history = self._check_history(y_history, design_history)

        if not fused:
            # One call batched over timesteps beats looping over T
            # unless b*T is large (due to materialisation costs).
            # max_folded_steps is a heuristic threshold determining switching.
            if (
                self.memoryless
                and self._n_steps(y_history, design_history, theta)
                <= self.max_folded_steps
            ):
                return self._memoryless_history_logp(y_history, design_history, theta)
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

    def _n_steps(self, y_history, design_history, theta):
        """Counts the steps to score: the broadcast batch size times T."""
        T = y_history.shape[-(len(self.event_shape_y) + 1)]
        batch_shape = torch.broadcast_shapes(
            y_history.shape[: y_history.ndim - 1 - len(self.event_shape_y)],
            design_history.shape[
                : design_history.ndim - 1 - len(self.event_shape_design)
            ],
            theta.shape[: theta.ndim - len(self.event_shape_theta)],
        )
        return batch_shape.numel() * T

    def _memoryless_history_logp(self, y_history, design_history, theta):
        r"""Computes logprob for a rollout with T folded into the batch.

        For efficient calculation on memoryless tasks: one ``_outcome_logp``
        call over ``[b_flat * T, ...]`` replaces the per-step loop.

        Reached through ``history_logp``: compiled when ``fused``, and eagerly
        for at most :attr:`max_folded_steps` steps, since it materializes all
        ``[b_flat * T, ...]`` entries at once.
        """
        T = y_history.shape[-(len(self.event_shape_y) + 1)]
        (y_flat, design_flat, theta_flat), batch_shape = broadcast_and_flatten(
            (y_history, (T, *self.event_shape_y)),
            (design_history, (T, *self.event_shape_design)),
            (theta, tuple(self.event_shape_theta)),
        )

        # theta is shared across the steps, which are folded into the batch
        b_flat = theta_flat.shape[0]
        theta_flat = theta_flat.unsqueeze(1).expand(b_flat, T, *self.event_shape_theta)
        logp = self._outcome_logp(
            y_flat.reshape(b_flat * T, *self.event_shape_y),
            theta_flat.reshape(b_flat * T, *self.event_shape_theta),
            design_flat.reshape(b_flat * T, *self.event_shape_design),
            y_history=None,
            design_history=None,
        )
        return logp.reshape(b_flat, T).sum(-1).reshape(batch_shape)

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
        y_context, design_context = self._context_history(y_history, design_history)

        batch_shape = torch.broadcast_shapes(
            design.shape[: design.ndim - len(self.event_shape_design)],
            y_context.shape[: y_context.ndim - len(self.event_shape_y) - 1],
            design_context.shape[
                : design_context.ndim - len(self.event_shape_design) - 1
            ],
        )
        # Keep the prior draw flat to preserve seeded sampling across priors.
        shape = sample_shape + batch_shape
        theta = self.sample_current_prior(shape.numel()).reshape(
            *shape, *self.event_shape_theta
        )
        y = self.sample_outcome(
            theta, design, y_history=y_history, design_history=design_history
        )
        return y, theta

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
            new_prior (Distribution or torch.distributions.Distribution):
                Prior to update to; a torch distribution is wrapped in
                :class:`~testbed.core.distributions.TorchDistribution`. Must
                have the same ``event_shape`` as the original prior and an
                empty ``batch_shape``, and its ``sample`` and ``log_prob`` must
                honour them; this is checked on one draw, as in ``__init__``.
        """
        new_prior = as_distribution(new_prior)
        self._validate_prior(new_prior)
        self._current_prior = new_prior
