"""Tests for the variational-posterior (Barber-Agakov) EIG estimator."""

import json
from functools import partial
from pathlib import Path

import pytest
import torch
from torch.distributions import HalfNormal, Independent, MultivariateNormal, Normal
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    ExpTransform,
    SigmoidTransform,
    StickBreakingTransform,
)

from testbed.core import RandomScaleGaussianDesignDistribution, Rollout
from testbed.core.design_spaces import BoxDesignSpace
from testbed.core.hooks import RolloutHook
from testbed.core.util import sort_by_first_coordinate
from testbed.design_nets import (
    AttentiveVariationalPosteriorNet,
    BlockTransform,
    ConstrainedAttentiveVariationalPosteriorNet,
    DeepSetPolicyNet,
    DeepSetVariationalPosteriorNet,
    GaussianPolicyNet,
)
from testbed.design_nets.variational_posterior import (
    GaussianMixtureVariationalPosteriorNet,
    VariationalPosteriorNet,
)
from testbed.design_objectives import (
    BarberAgakov,
    BarberAgakovReinforce,
    MyopicBarberAgakov,
    posterior_nll,
)
from testbed.design_objectives.reinforce import leave_one_out_baseline
from testbed.designers import PolicyDesigner, RandomDesigner, SequentialDesigner
from testbed.tasks import (
    CES,
    LinearRegression,
    LocationFinding,
    MichaelisMenten,
    TemporalDiscounting,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_task():
    return TemporalDiscounting()


def _make_design_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _make_stochastic_net(task):
    space = BoxDesignSpace(low=[5.0, 20.0], high=[40.0, 80.0], event_shape=(2,))
    return GaussianPolicyNet(space, event_shape_y=task.event_shape_y)


def _make_posterior(task, n_components=3):
    return DeepSetVariationalPosteriorNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=n_components,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
    )


def _make_designer(task, policy_net, T=3, **kwargs):
    # PolicyDesigner defaults use_raw_history to False for every net; reproduce
    # the old per-net default unless a caller asks for something else.
    kwargs.setdefault("use_raw_history", not policy_net.produces_policy_log_probs)
    return PolicyDesigner(task, T=T, design_net=policy_net, **kwargs)


def _make_rollout(task, policy_net, b=8, T=3, **kwargs):
    designer = _make_designer(task, policy_net, T=T, **kwargs)
    theta0 = task.sample_prior(b)
    return designer.rollout(theta0)


def _policy_grad(net, gain):
    grads = torch.autograd.grad(
        gain, list(net.parameters()), retain_graph=True, allow_unused=True
    )
    return torch.cat(
        [
            (grad if grad is not None else torch.zeros_like(param)).flatten()
            for grad, param in zip(grads, net.parameters())
        ]
    )


class _ExactPosterior(VariationalPosteriorNet):
    """Test-only posterior returning the exact Gaussian-linear posterior.

    Only valid for LinearRegression(dim=1) with a single (T=1) observation and
    a fixed design; used to check BarberAgakov's bound is tight (matches the exact
    EIG) when q == p(theta | y, d).
    """

    def __init__(self, prior_loc, prior_cov, design_row, noise_scale):
        super().__init__(torch.Size([prior_loc.shape[-1]]))
        self.prior_loc = prior_loc
        self.prior_precision = torch.linalg.inv(prior_cov)
        self.design_row = design_row
        self.noise_scale = noise_scale

    def distribution(self, y_history, design_history, mask=None):
        y = y_history[:, 0, 0]  # [b], single scalar observation
        xd = self.design_row  # [D]
        post_precision = (
            self.prior_precision + torch.outer(xd, xd) / self.noise_scale**2
        )
        post_cov = torch.linalg.inv(post_precision)
        post_mean = (
            post_cov
            @ (
                self.prior_precision @ self.prior_loc
                + xd * y.unsqueeze(-1) / self.noise_scale**2
            ).T
        )
        post_mean = post_mean.T  # [b, D]
        return MultivariateNormal(loc=post_mean, covariance_matrix=post_cov)

    def log_prob(self, theta, y_history, design_history, mask=None):
        theta_flat = self._flatten_theta(theta)
        return self.distribution(y_history, design_history, mask=mask).log_prob(
            theta_flat
        )

    def sample(self, y_history, design_history, n_samples: int = 1, mask=None):
        dist = self.distribution(y_history, design_history, mask=mask)
        return dist.sample((n_samples,)).reshape(
            n_samples, y_history.shape[0], *self.event_shape_theta
        )


# ============================================================================
# VariationalPosteriorNet: shapes and mixture correctness
# ============================================================================


@pytest.mark.parametrize(
    "net_fn",
    [
        lambda task: DeepSetVariationalPosteriorNet(
            task.event_shape_y,
            task.event_shape_design,
            task.event_shape_theta,
            n_components=3,
            encoder_dim_h=[16],
            dim_emb=16,
            trunk_dim_h=16,
        ),
        lambda task: AttentiveVariationalPosteriorNet(
            task.event_shape_y,
            task.event_shape_design,
            task.event_shape_theta,
            n_components=3,
            encoder_dim_h=[16],
            dim_emb=16,
            num_heads=2,
            num_blocks=1,
        ),
    ],
)
def test_posterior_log_prob_shape_and_finite_1d_theta(net_fn):
    task = _make_task()  # HTD: event_shape_theta == [2]
    posterior = net_fn(task)
    rollout = _make_rollout(task, _make_design_net(task))

    log_q = posterior.log_prob(rollout.theta, rollout.y_history, rollout.design_history)
    assert log_q.shape == (rollout.b,)
    assert torch.isfinite(log_q).all()


def test_posterior_log_prob_multidim_theta_flattens_correctly():
    task = LocationFinding(n_sources=2, dim_theta=2)  # event_shape_theta == [2, 2]
    posterior = _make_posterior(task)
    designer = RandomDesigner(task, T=3)
    theta0 = task.sample_prior(6)
    rollout = designer.rollout(theta0)

    log_q = posterior.log_prob(rollout.theta, rollout.y_history, rollout.design_history)
    assert log_q.shape == (6,)
    assert torch.isfinite(log_q).all()


def test_posterior_log_prob_rejects_wrong_theta_shape():
    task = _make_task()
    posterior = _make_posterior(task)
    rollout = _make_rollout(task, _make_design_net(task))
    with pytest.raises(ValueError):
        posterior.log_prob(
            rollout.theta[:, :1], rollout.y_history, rollout.design_history
        )


def test_posterior_sample_shapes():
    task = _make_task()
    posterior = _make_posterior(task)
    rollout = _make_rollout(task, _make_design_net(task), b=5)

    s1 = posterior.sample(rollout.y_history, rollout.design_history, n_samples=1)
    s3 = posterior.sample(rollout.y_history, rollout.design_history, n_samples=3)
    assert s1.shape == (
        1,
        5,
        *task.event_shape_theta,
    )  # leading n_samples dim always kept
    assert s3.shape == (3, 5, *task.event_shape_theta)


def test_mixture_with_one_component_matches_independent_normal():
    torch.manual_seed(0)
    task = _make_task()
    posterior = DeepSetVariationalPosteriorNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=1,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
    )
    rollout = _make_rollout(task, _make_design_net(task), b=4)

    # .dist reaches the wrapped torch mixture, whose internals this test checks
    dist = posterior.distribution(rollout.y_history, rollout.design_history).dist
    loc = dist.component_distribution.base_dist.loc[:, 0]
    scale = dist.component_distribution.base_dist.scale[:, 0]
    reference = Independent(Normal(loc, scale), 1)

    theta_flat = rollout.theta.reshape(rollout.b, posterior.dim_theta)
    torch.testing.assert_close(
        dist.log_prob(theta_flat), reference.log_prob(theta_flat)
    )


def test_mixture_with_three_components_matches_manual_logsumexp():
    torch.manual_seed(1)
    task = _make_task()
    posterior = _make_posterior(task, n_components=3)
    rollout = _make_rollout(task, _make_design_net(task), b=4)

    dist = posterior.distribution(rollout.y_history, rollout.design_history).dist
    theta_flat = rollout.theta.reshape(rollout.b, posterior.dim_theta)

    loc = dist.component_distribution.base_dist.loc  # [b, K, D]
    scale = dist.component_distribution.base_dist.scale
    logits = dist.mixture_distribution.logits  # [b, K]

    comp_logp = Normal(loc, scale).log_prob(theta_flat.unsqueeze(1)).sum(-1)  # [b, K]
    manual = torch.logsumexp(torch.log_softmax(logits, dim=-1) + comp_logp, dim=-1)

    torch.testing.assert_close(dist.log_prob(theta_flat), manual)


# ============================================================================
# BarberAgakov: bound correctness against a closed-form EIG
# ============================================================================


def _linear_regression_setup(x_val=0.7, noise_scale=0.5):
    task = LinearRegression(dim=1, noise_scale=noise_scale)
    prior_loc = task.prior.loc
    prior_cov = task.prior.cov
    design_row = torch.tensor([1.0, x_val])  # [1, x]

    post_precision = (
        torch.linalg.inv(prior_cov)
        + torch.outer(design_row, design_row) / noise_scale**2
    )
    post_cov = torch.linalg.inv(post_precision)
    exact_eig = 0.5 * (
        torch.linalg.slogdet(prior_cov)[1] - torch.linalg.slogdet(post_cov)[1]
    )
    return task, prior_loc, prior_cov, design_row, exact_eig


def _fixed_design_rollout(task, x_val, b):
    theta0 = task.sample_prior(b)
    design = torch.tensor([x_val]).expand(b, 1)  # [b, *event_shape_design]
    y = task.sample_outcome(theta0, design)  # [b, *event_shape_y]
    from testbed.core.rollout import Rollout

    return Rollout(
        theta=theta0, y_history=y.unsqueeze(1), design_history=design.unsqueeze(1)
    )


def test_untrained_posterior_gives_a_bound_strictly_below_exact_eig():
    torch.manual_seed(0)
    x_val = 0.7
    task, *_rest, exact_eig = _linear_regression_setup(x_val=x_val)
    rollout = _fixed_design_rollout(task, x_val, b=8192)

    posterior = DeepSetVariationalPosteriorNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=3,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
    )
    bound = BarberAgakov(task, posterior).estimate_gain(rollout)

    assert (
        bound.item() < exact_eig.item() - 0.5
    )  # comfortably below, well outside MC noise


def test_exact_posterior_bound_matches_exact_eig():
    torch.manual_seed(1)
    x_val = 0.7
    task, prior_loc, prior_cov, design_row, exact_eig = _linear_regression_setup(
        x_val=x_val
    )
    rollout = _fixed_design_rollout(task, x_val, b=32768)

    posterior = _ExactPosterior(prior_loc, prior_cov, design_row, task.noise_scale)
    bound = BarberAgakov(task, posterior).estimate_gain(rollout)

    assert bound.item() == pytest.approx(exact_eig.item(), abs=0.05)


# ============================================================================
# BarberAgakov: gradient flow and freezing
# ============================================================================


def test_var_post_gradient_reaches_posterior_params():
    task = _make_task()
    posterior = _make_posterior(task)
    rollout = _make_rollout(task, _make_design_net(task))

    gain = BarberAgakov(task, posterior).estimate_gain(rollout)
    gain.backward()

    trainable = [p for n, p in posterior.named_parameters() if n != "encoder.null_enc"]
    assert all(p.grad is not None for p in trainable)
    assert any(p.grad.abs().sum() > 0 for p in trainable)


def test_var_post_gradient_reaches_deterministic_policy_and_posterior_in_one_backward():
    task = _make_task()
    posterior = _make_posterior(task)
    policy_net = _make_design_net(task)
    designer = _make_designer(task, policy_net)
    rollout = designer.rollout(task.sample_prior(8))

    gain = BarberAgakovReinforce(
        task, posterior, stochastic_design_policy=False
    ).estimate_gain(rollout, train_posterior=True)
    gain.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy_net.parameters()
    )
    trainable = [p for n, p in posterior.named_parameters() if n != "encoder.null_enc"]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)


def test_var_post_reinforce_train_posterior_false_blocks_posterior_gradient_even_if_unfrozen():
    """train_posterior=False must guarantee zero posterior gradient in the
    deterministic/pathwise branch too, regardless of the posterior's own
    requires_grad -- not merely rely on the caller having frozen it."""
    task = _make_task()
    posterior = _make_posterior(task)
    policy_net = _make_design_net(task)
    designer = _make_designer(task, policy_net)
    rollout = designer.rollout(task.sample_prior(8))
    objective = BarberAgakovReinforce(task, posterior, stochastic_design_policy=False)

    gain = objective.estimate_gain(rollout, train_posterior=False)
    gain.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy_net.parameters()
    )
    trainable = [p for n, p in posterior.named_parameters() if n != "encoder.null_enc"]
    assert all(p.grad is None for p in trainable)
    # requires_grad is restored afterward, not permanently disabled
    assert all(p.requires_grad for p in posterior.parameters())


def test_frozen_posterior_receives_no_gradient():
    task = _make_task()
    posterior = _make_posterior(task)
    posterior.eval()
    for p in posterior.parameters():
        p.requires_grad_(False)

    rollout = _make_rollout(task, _make_design_net(task))
    gain = BarberAgakov(task, posterior).estimate_gain(rollout)
    gain.backward()

    assert all(p.grad is None for p in posterior.parameters())


def test_var_post_raises_on_non_evaluable_prior():
    task = _make_task()
    posterior = _make_posterior(task)
    task.prior.log_prob = lambda *_: (_ for _ in ()).throw(NotImplementedError())
    with pytest.raises(ValueError):
        BarberAgakov(task, posterior)


# ============================================================================
# BarberAgakovReinforce: matches BarberAgakov forward value, and DAD-Eq.16-style gradient
# ============================================================================


def test_var_post_reinforce_forward_value_matches_var_post():
    task = _make_task()
    posterior = _make_posterior(task)
    rollout = _make_rollout(task, _make_design_net(task))

    torch.manual_seed(2)
    reinforce_gain = BarberAgakovReinforce(
        task, posterior, stochastic_design_policy=False
    ).estimate_gain(rollout)
    torch.manual_seed(2)
    plain_gain = BarberAgakov(task, posterior).estimate_gain(rollout)

    assert torch.allclose(reinforce_gain, plain_gain)


@pytest.mark.parametrize("baseline", [None, "leave_one_out"])
def test_deterministic_reinforce_matches_dad_eq16_reference(baseline):
    torch.manual_seed(0)
    task = _make_task()
    posterior = _make_posterior(task)
    policy_net = _make_design_net(task)
    designer = _make_designer(task, policy_net)
    rollout = designer.rollout(task.sample_prior(8))

    barber_agakov = BarberAgakov(task, posterior)
    reinforce = BarberAgakovReinforce(
        task, posterior, stochastic_design_policy=False, baseline=baseline
    )

    torch.manual_seed(1)
    gain = reinforce.estimate_gain(rollout)

    torch.manual_seed(1)
    reward = barber_agakov.per_sample_gain(rollout)
    baseline_value = (
        leave_one_out_baseline(reward.detach()) if baseline == "leave_one_out" else 0.0
    )
    score = task.history_logp(rollout.y_history, rollout.design_history, rollout.theta)
    reference = (
        reward + (reward.detach() - baseline_value) * (score - score.detach())
    ).mean()

    assert torch.allclose(gain, reference)
    assert torch.allclose(
        _policy_grad(policy_net, gain), _policy_grad(policy_net, reference)
    )


def test_stochastic_constant_value_loo_baseline_zero_grad():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    designer = _make_designer(task, policy_net)
    rollout = designer.rollout(task.sample_prior(8))

    posterior = _make_posterior(task)

    class _ConstantBarberAgakov(BarberAgakov):
        def per_sample_gain(self, rollout):
            return rollout.y_history.new_full((rollout.b,), 3.0)

    reinforce = BarberAgakovReinforce(
        task, posterior, stochastic_design_policy=True, baseline="leave_one_out"
    )
    reinforce.barber_agakov = _ConstantBarberAgakov(task, posterior)

    gain = reinforce.estimate_gain(rollout)
    grad = _policy_grad(policy_net, gain)

    assert gain.item() == pytest.approx(3.0)
    assert torch.equal(grad, torch.zeros_like(grad))


# ============================================================================
# Joint-training gradient split (mirrors methods/var_post.Trainer.train_step)
# ============================================================================


def test_estimate_gain_train_posterior_updates_both_nets_in_one_backward_stochastic():
    """A single estimate_gain(train_posterior=True) + one backward() must update
    both design_net (via the REINFORCE surrogate) and posterior_net (via the
    injected term) for a stochastic, non-reparameterized policy."""
    task = _make_task()
    posterior = _make_posterior(task)
    policy_net = _make_stochastic_net(task)
    designer = _make_designer(task, policy_net, reparameterize_policy=False)
    rollout = designer.rollout(task.sample_prior(8))
    assert "policy_log_probs" in rollout.extra

    objective = BarberAgakovReinforce(task, posterior, stochastic_design_policy=True)

    gain = objective.estimate_gain(rollout, train_posterior=True)
    gain.backward()

    trainable_posterior = [
        p for n, p in posterior.named_parameters() if n != "encoder.null_enc"
    ]
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy_net.parameters()
    )
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in trainable_posterior
    )


def test_train_posterior_preserves_the_interpretable_gain_value():
    """The injected posterior term must be zero-forward-value: estimate_gain's
    returned number (used for logging / best-checkpoint comparisons) should
    be unaffected by train_posterior, only its gradient."""
    task = _make_task()
    posterior = _make_posterior(task)
    policy_net = _make_stochastic_net(task)
    designer = _make_designer(task, policy_net, reparameterize_policy=False)
    rollout = designer.rollout(task.sample_prior(8))
    objective = BarberAgakovReinforce(task, posterior, stochastic_design_policy=True)

    plain = objective.estimate_gain(rollout, train_posterior=False)
    with_posterior = objective.estimate_gain(rollout, train_posterior=True)
    torch.testing.assert_close(plain, with_posterior)


def test_train_posterior_does_not_affect_policy_gradient_deterministic():
    """train_posterior only gates posterior_net's gradient; design_net's gradient
    must be identical either way in the deterministic/pathwise branch."""
    task = _make_task()
    posterior = _make_posterior(task)
    policy_net = _make_design_net(task)
    designer = _make_designer(task, policy_net)
    rollout = designer.rollout(task.sample_prior(8))
    objective = BarberAgakovReinforce(task, posterior, stochastic_design_policy=False)

    plain = objective.estimate_gain(rollout, train_posterior=False)
    with_posterior = objective.estimate_gain(rollout, train_posterior=True)
    torch.testing.assert_close(plain, with_posterior)

    grad_plain = _policy_grad(policy_net, plain)
    grad_with_posterior = _policy_grad(policy_net, with_posterior)
    torch.testing.assert_close(grad_plain, grad_with_posterior)


def test_posterior_gain_method_matches_var_post_estimate_gain():
    task = _make_task()
    posterior = _make_posterior(task)
    rollout = _make_rollout(task, _make_design_net(task))
    objective = BarberAgakovReinforce(task, posterior, stochastic_design_policy=False)

    torch.testing.assert_close(
        objective.posterior_gain(rollout),
        objective.barber_agakov.estimate_gain(rollout),
    )


# ============================================================================
# Trainer smoke tests
# ============================================================================


def _make_trainer(joint):
    from types import SimpleNamespace

    from testbed.methods.var_post import Trainer

    task = _make_task()
    posterior = _make_posterior(task)
    design_net = _make_design_net(task)
    designer = _make_designer(task, design_net)
    pretrain_designer = RandomDesigner(task, T=3)
    objective = BarberAgakovReinforce(task, posterior, stochastic_design_policy=False)
    posterior_loss = partial(posterior_nll, posterior)

    cfg = SimpleNamespace(
        joint=joint,
        n_posterior_outer=8,
        n_outer=8,
        posterior_grad_clip=1.0,
        grad_clip=1.0,
        objective=SimpleNamespace(requires_policy_log_probs=False),
    )
    trainer = Trainer(
        task=task,
        designer=designer,
        posterior_net=posterior,
        objective=objective,
        posterior_loss=posterior_loss,
        policy_optimizer=torch.optim.Adam(design_net.parameters(), lr=1e-3),
        posterior_optimizer=torch.optim.Adam(posterior.parameters(), lr=1e-3),
        cfg=cfg,
        pretrain_designer=pretrain_designer,
    )
    return trainer, design_net, posterior


def _snapshot(net):
    return [p.detach().clone() for p in net.parameters()]


def _any_changed(before, net):
    return any(not torch.equal(b, p) for b, p in zip(before, net.parameters()))


def test_posterior_pretrain_step_changes_only_posterior():
    trainer, design_net, posterior = _make_trainer(joint=False)
    design_before, posterior_before = _snapshot(design_net), _snapshot(posterior)

    metrics = trainer.posterior_pretrain_step()

    assert torch.isfinite(torch.tensor(metrics["posterior_pretrain/loss"]))
    assert not _any_changed(design_before, design_net)
    assert _any_changed(posterior_before, posterior)


def test_frozen_train_step_changes_only_design_net():
    trainer, design_net, posterior = _make_trainer(joint=False)
    trainer._freeze_posterior()
    design_before, posterior_before = _snapshot(design_net), _snapshot(posterior)

    metrics = trainer.train_step()

    assert torch.isfinite(torch.tensor(metrics["gain"]))
    assert _any_changed(design_before, design_net)
    assert not _any_changed(posterior_before, posterior)


def test_joint_train_step_changes_both_nets():
    trainer, design_net, posterior = _make_trainer(joint=True)
    design_before, posterior_before = _snapshot(design_net), _snapshot(posterior)

    metrics = trainer.train_step()

    assert torch.isfinite(torch.tensor(metrics["gain"]))
    assert _any_changed(design_before, design_net)
    assert _any_changed(posterior_before, posterior)


# ============================================================================
# Two-phase resume
#
# The budget and checkpointing machinery itself is ActionBED's. What is new
# here is that var_post's loop spans two phases with one step counter, and
# that the posterior freeze lives between them.
# ============================================================================


RECIPES_DIR = str(Path(__file__).resolve().parents[1] / "testbed" / "recipes")


def _budget_cfg(**overrides):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base="1.3", config_dir=RECIPES_DIR, job_name="var_post_budget"
    ):
        cfg = compose(config_name="method/var_post")
    cfg.device = "cpu"
    cfg.task.n_sources = 1
    cfg.objective.fused = False
    cfg.n_posterior_pretrain_iter = 2
    cfg.n_train_iter = 3
    cfg.n_outer = 8
    cfg.n_posterior_outer = 8
    cfg.eval_every = 0
    cfg.checkpoint_every = 0
    cfg.ema_decay = 0.99
    cfg.policy_cosine_alpha = 0.1
    cfg.posterior_warmup_frac = 0.4
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _budget_manager(tmp_path, name):
    from types import SimpleNamespace

    from testbed.core.manager import Manager

    return Manager(
        SimpleNamespace(), root=str(tmp_path), name=name, timestamp=False, console=False
    )


def _run(cfg, tmp_path, name, clock=None, monkeypatch=None):
    from hydra.utils import instantiate

    from testbed.methods import var_post as module

    trainer = module.build_trainer(
        cfg, instantiate(cfg.task).to("cpu"), _budget_manager(tmp_path, name)
    )
    if clock is not None:
        values = iter(clock)
        last = [clock[0]]

        def monotonic():
            last[0] = next(values, last[0])
            return last[0]

        monkeypatch.setattr(module.time, "monotonic", monotonic)
    trainer.train()
    if clock is not None:
        monkeypatch.undo()
    return trainer


@pytest.mark.parametrize("stop_after", [1, 3])
def test_resume_across_both_phases_matches_an_uninterrupted_run(
    tmp_path, monkeypatch, stop_after
):
    """Stopping inside pretraining or inside policy training must both resume cleanly."""
    from testbed.methods.var_post import load_checkpoint

    whole = _run(_budget_cfg(), tmp_path, "whole")

    # One read as train() starts, then one per step check; trip on the last.
    clock = [0.0] * stop_after + [99.0]
    stopped = _run(
        _budget_cfg(train_time_budget_seconds=50.0),
        tmp_path,
        "part",
        clock=clock,
        monkeypatch=monkeypatch,
    )
    assert stopped.step == stop_after - 1
    assert stopped.stop_reason == "time_budget"

    resumed = load_checkpoint(
        tmp_path / "part" / "final_step.pt",
        train_time_budget_seconds=float("inf"),
        manager=_budget_manager(tmp_path, "rest"),
    )
    resumed.train()

    assert resumed.step == whole.step
    for attr in ("designer.design_net", "posterior_net", "design_net_ema"):
        target = resumed
        source = whole
        for part in attr.split("."):
            target, source = getattr(target, part), getattr(source, part)
        expected, actual = source.state_dict(), target.state_dict()
        assert all(torch.equal(expected[k], actual[k]) for k in expected), attr
    assert (
        resumed.policy_scheduler.get_last_lr() == whole.policy_scheduler.get_last_lr()
    )
    assert (
        resumed.posterior_scheduler.get_last_lr()
        == whole.posterior_scheduler.get_last_lr()
    )


def test_resume_into_the_policy_phase_refreezes_the_posterior(tmp_path, monkeypatch):
    """requires_grad is in no state dict, so train() has to re-apply the freeze."""
    from testbed.methods.var_post import load_checkpoint

    cfg = _budget_cfg(joint=False, train_time_budget_seconds=50.0)
    # Stop on step 2, the first one past the two pretraining steps.
    _run(cfg, tmp_path, "frozen", clock=[0.0, 0.0, 0.0, 99.0], monkeypatch=monkeypatch)

    resumed = load_checkpoint(
        tmp_path / "frozen" / "final_step.pt",
        train_time_budget_seconds=float("inf"),
        manager=_budget_manager(tmp_path, "frozen_rest"),
    )
    posterior_before = _snapshot(resumed.posterior_net)
    resumed.train()

    assert not resumed.posterior_net.training
    assert all(not p.requires_grad for p in resumed.posterior_net.parameters())
    assert not _any_changed(posterior_before, resumed.posterior_net)


def _last_metrics(tmp_path, name):
    return json.loads((tmp_path / name / "metrics.jsonl").read_text().splitlines()[-1])


def test_train_logs_how_far_it_got(tmp_path, monkeypatch):
    """A time-stopped run has to say how far it got, in both phases' terms.

    The step counter spans pretraining and policy training, so the raw count is
    not comparable with methods that have only one phase.
    """
    cfg = _budget_cfg()
    _run(cfg, tmp_path, "whole")

    whole = _last_metrics(tmp_path, "whole")
    total = cfg.n_posterior_pretrain_iter + cfg.n_train_iter
    assert whole["train/steps_completed"] == total
    assert whole["train/policy_steps_completed"] == cfg.n_train_iter
    assert whole["train/stop_reason"] == "iteration_limit"
    assert whole["step"] == total - 1

    # Stop on step 2, the first one past the two pretraining steps.
    _run(
        _budget_cfg(train_time_budget_seconds=50.0),
        tmp_path,
        "part",
        clock=[0.0, 0.0, 0.0, 99.0],
        monkeypatch=monkeypatch,
    )

    part = _last_metrics(tmp_path, "part")
    assert part["train/steps_completed"] == 3
    assert part["train/policy_steps_completed"] == 1
    assert part["train/stop_reason"] == "time_budget"


def test_pretraining_only_run_reports_no_policy_steps(tmp_path, monkeypatch):
    """Stopping inside pretraining must floor the policy count at zero, not go negative."""
    _run(
        _budget_cfg(train_time_budget_seconds=50.0),
        tmp_path,
        "early",
        clock=[0.0, 99.0],
        monkeypatch=monkeypatch,
    )

    early = _last_metrics(tmp_path, "early")
    assert early["train/steps_completed"] == 1
    assert early["train/policy_steps_completed"] == 0


# ============================================================================
# RandomScaleGaussianDesignDistribution
# ============================================================================


def test_random_scale_gaussian_shapes_and_log_prob_finite():
    task = LocationFinding(n_sources=1, dim_theta=2)
    T = 5
    dist = RandomScaleGaussianDesignDistribution(T, task.event_shape_design)
    designer = RandomDesigner(task, T, dist=dist)

    rollout = designer.rollout(task.sample_prior(6))
    assert rollout.design_history.shape == (6, T, *task.event_shape_design)
    assert torch.isfinite(rollout.extra["policy_log_probs"]).all()


def test_random_scale_gaussian_shares_scale_within_a_trajectory():
    # Checks the actual grouping mechanism directly (the cached scale used to
    # both draw the noise and answer log_prob) rather than trying to infer it
    # statistically from noisy sampled designs.
    torch.manual_seed(0)
    T = 4
    dist = RandomScaleGaussianDesignDistribution(T, event_shape_design=(2,))

    n_trajectories = 5
    dist.sample(n_trajectories * T)
    scale_blocks = dist._last_scale_flat.reshape(n_trajectories, T)

    # every one of the T entries within a trajectory shares its scale...
    assert torch.equal(scale_blocks, scale_blocks[:, :1].expand(n_trajectories, T))
    # ...while different trajectories get (almost surely) different scales.
    assert scale_blocks[:, 0].unique().numel() == n_trajectories


def test_random_scale_gaussian_rejects_non_multiple_of_t():
    task = LocationFinding(n_sources=1, dim_theta=2)
    T = 5
    dist = RandomScaleGaussianDesignDistribution(T, task.event_shape_design)

    class _NoOpHook(RolloutHook):
        pass

    designer = RandomDesigner(task, T, dist=dist, hooks=[_NoOpHook()])
    with pytest.raises(ValueError):
        designer.rollout(task.sample_prior(6))


def test_var_post_recipe_pretrain_designer_uses_random_scale_gaussian():
    from types import SimpleNamespace

    from testbed.methods.var_post import build_pretrain_designer

    task = LocationFinding(n_sources=1, dim_theta=2)
    # hydra.utils.instantiate requires a dict/DictConfig for cfg.pretrain_design_distr
    # itself (not a SimpleNamespace); cfg's own attribute access can still be
    # a SimpleNamespace, matching how the Trainer's other tests build cfg.
    cfg = SimpleNamespace(
        T=5,
        pretrain_design_distr={
            "_target_": "testbed.core.RandomScaleGaussianDesignDistribution",
            "scale_low": 0.5,
            "scale_high": 2.0,
        },
    )
    designer = build_pretrain_designer(cfg, task)
    assert isinstance(designer.dist, RandomScaleGaussianDesignDistribution)

    rollout = designer.rollout(task.sample_prior(4))
    assert rollout.design_history.shape == (4, 5, *task.event_shape_design)


# ============================================================================
# canonicalize_theta (multi-source label-switching symmetry)
# ============================================================================


def test_sort_by_first_coordinate_matches_manual_reference():
    torch.manual_seed(0)
    x = torch.randn(5, 3, 2)  # [batch, n_objects, object_shape]
    sorted_x = sort_by_first_coordinate(x)

    assert torch.all(sorted_x[:, :-1, 0] <= sorted_x[:, 1:, 0])
    for i in range(5):
        # every row is some permutation of the original objects, not new values
        assert torch.equal(
            sorted_x[i][sorted_x[i, :, 0].argsort().argsort()].sort(dim=0).values,
            x[i].sort(dim=0).values,
        )


def _make_location_posterior(task, canonicalize_theta=False):
    return DeepSetVariationalPosteriorNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=3,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
        canonicalize_theta=canonicalize_theta,
    )


def test_canonicalize_theta_makes_var_post_invariant_to_source_order():
    task = LocationFinding(n_sources=2, dim_theta=2)
    posterior = _make_location_posterior(task)
    designer = RandomDesigner(task, T=3)
    theta0 = task.sample_prior(6)
    rollout = designer.rollout(theta0)

    # swap the two sources' order in theta only (the likelihood/y_history is
    # unaffected either way, since it's exactly permutation-invariant)
    swapped_rollout = Rollout(
        theta=rollout.theta.flip(dims=(1,)),
        y_history=rollout.y_history,
        design_history=rollout.design_history,
    )

    barber_agakov_plain = BarberAgakov(task, posterior)
    gain = barber_agakov_plain.per_sample_gain(rollout)
    gain_swapped = barber_agakov_plain.per_sample_gain(swapped_rollout)
    assert not torch.allclose(
        gain, gain_swapped
    )  # generically different without canonicalization

    # canonicalization lives on the net, so the objective needs no flag
    canon_posterior = _make_location_posterior(task, canonicalize_theta=True)
    barber_agakov_canon = BarberAgakov(task, canon_posterior)
    gain_canon = barber_agakov_canon.per_sample_gain(rollout)
    gain_canon_swapped = barber_agakov_canon.per_sample_gain(swapped_rollout)
    torch.testing.assert_close(
        gain_canon, gain_canon_swapped
    )  # identical with canonicalization


def test_posterior_nll_inherits_the_nets_canonicalization():
    task = LocationFinding(n_sources=2, dim_theta=2)
    posterior = _make_location_posterior(task, canonicalize_theta=True)
    plain = _make_location_posterior(task)
    plain.load_state_dict(posterior.state_dict())
    designer = RandomDesigner(task, T=3)
    rollout = designer.rollout(task.sample_prior(6))

    loss = posterior_nll(posterior, rollout, reduction="none")
    # the same weights without canonicalization, fed pre-sorted theta by hand
    manual = -plain.log_prob(
        sort_by_first_coordinate(rollout.theta),
        rollout.y_history,
        rollout.design_history,
    )
    torch.testing.assert_close(loss, manual)


def test_canonicalize_theta_requires_an_object_axis():
    task = TemporalDiscounting()  # event_shape_theta == [2]
    with pytest.raises(ValueError, match="exchangeable object axis"):
        DeepSetVariationalPosteriorNet(
            task.event_shape_y,
            task.event_shape_design,
            task.event_shape_theta,
            n_components=3,
            encoder_dim_h=[16],
            dim_emb=16,
            trunk_dim_h=16,
            canonicalize_theta=True,
        )


def test_canonicalized_net_sorts_its_samples():
    task = LocationFinding(n_sources=2, dim_theta=2)
    posterior = _make_location_posterior(task, canonicalize_theta=True)
    designer = RandomDesigner(task, T=3)
    rollout = designer.rollout(task.sample_prior(6))

    samples = posterior.sample(rollout.y_history, rollout.design_history, n_samples=4)
    assert samples.shape == (4, 6, *task.event_shape_theta)
    assert torch.all(samples[..., :-1, 0] <= samples[..., 1:, 0])


# ============================================================================
# MyopicBarberAgakov (single-design DesignObjective variant)
# ============================================================================


def test_myopic_bound_with_exact_posterior_matches_exact_eig():
    torch.manual_seed(1)
    x_val = 0.7
    task, prior_loc, prior_cov, design_row, exact_eig = _linear_regression_setup(
        x_val=x_val
    )
    posterior = _ExactPosterior(prior_loc, prior_cov, design_row, task.noise_scale)

    # the task design is x alone; design_row is the regressor [1, x]
    objective = MyopicBarberAgakov(task, posterior, n_outer=32768)
    bound = objective.estimate_gain(torch.tensor([x_val]))

    assert bound.item() == pytest.approx(exact_eig.item(), abs=0.05)


def test_myopic_gain_reaches_both_the_design_and_the_posterior():
    torch.manual_seed(0)
    task = LinearRegression(dim=2)
    posterior = DeepSetVariationalPosteriorNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=3,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
    )
    design = task.design_space.sample().requires_grad_(True)

    gain = MyopicBarberAgakov(task, posterior, n_outer=64).estimate_gain(design)
    assert gain.ndim == 0
    gain.backward()

    assert design.grad is not None and torch.isfinite(design.grad).all()
    trainable = [p for n, p in posterior.named_parameters() if n != "encoder.null_enc"]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)


def test_myopic_objective_rejects_a_batch_of_designs():
    task = LinearRegression(dim=2)
    posterior = DeepSetVariationalPosteriorNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=2,
        encoder_dim_h=[8],
        dim_emb=8,
        trunk_dim_h=8,
    )
    objective = MyopicBarberAgakov(task, posterior, n_outer=8)
    with pytest.raises(ValueError, match="single design"):
        objective.estimate_gain(task.design_space.sample(batch_shape=(3,)))


def test_myopic_objective_drives_a_sequential_designer():
    torch.manual_seed(0)
    task = LinearRegression(dim=2)
    posterior = DeepSetVariationalPosteriorNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=2,
        encoder_dim_h=[8],
        dim_emb=8,
        trunk_dim_h=8,
    )
    designer = SequentialDesigner(
        task,
        T=2,
        objective=MyopicBarberAgakov(task, posterior, n_outer=32),
        lr=1e-2,
        n_step_opt=2,
        update_posterior=False,
    )
    rollout = designer.rollout(task.sample_prior(1))

    assert rollout.design_history.shape == (1, 2, *task.event_shape_design)
    assert torch.isfinite(rollout.design_history).all()


# ============================================================================
# Constrained (reparametrised) posteriors
# ============================================================================


def _ces_transform():
    """The CES map: 4 unconstrained coordinates -> (rho, alpha_1..3, u).

    Mirrors the transform in experiments/ces/joint_var_post.yaml, including
    the 0.01 rho floor that matches CESPrior.sample's squash.
    """
    return BlockTransform(
        transforms=[
            ComposeTransform(
                [SigmoidTransform(), AffineTransform(loc=0.01, scale=0.99)]
            ),
            StickBreakingTransform(),
            ExpTransform(),
        ],
        raw_sizes=[1, 2, 1],
        theta_sizes=[1, 3, 1],
    )


def _make_ces_posterior(task, n_components=3):
    return ConstrainedAttentiveVariationalPosteriorNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_theta=task.event_shape_theta,
        transform=_ces_transform(),
        n_components=n_components,
        encoder_dim_h=[16],
        dim_emb=16,
        num_heads=2,
        num_blocks=1,
        trunk_dim_h=16,
    )


def _ces_history(task, b=16, T=3):
    return torch.rand(b, T, 1), torch.rand(b, T, 6) * 100


def test_block_transform_round_trips_on_real_prior_draws():
    torch.manual_seed(0)
    transform = _ces_transform()

    raw = torch.randn(64, 4)
    assert torch.allclose(transform.inv(transform(raw)), raw, atol=1e-5)

    theta = CES().sample_prior(2_000)
    assert torch.allclose(transform(transform.inv(theta)), theta, atol=1e-5)
    assert torch.isfinite(transform.inv(theta)).all()


def test_block_transform_output_satisfies_the_ces_constraints():
    torch.manual_seed(0)
    theta = _ces_transform()(torch.randn(128, 4))

    assert ((theta[:, 0] > 0.01) & (theta[:, 0] < 1.0)).all()
    assert torch.allclose(theta[:, 1:4].sum(-1), torch.ones(128), atol=1e-6)
    assert (theta[:, 1:4] > 0).all()
    assert (theta[:, 4] > 0).all()


def test_block_transform_log_det_matches_the_square_jacobian():
    # The reference is the Jacobian with alpha's redundant third coordinate
    # dropped, which is the simplex's coordinate measure d(alpha_1) d(alpha_2)
    # and the convention Dirichlet.log_prob uses. NOT the Gram determinant
    # 0.5 * logdet(J^T J) of the full [5, 4] Jacobian: that is the embedded
    # simplex's 2-dim Hausdorff measure and is larger by exactly log(sqrt(3)).
    torch.manual_seed(0)
    transform = _ces_transform()
    keep = [0, 1, 2, 4]  # rho, alpha_1, alpha_2, u

    for _ in range(4):
        raw = torch.randn(4, dtype=torch.float64)
        jacobian = torch.autograd.functional.jacobian(
            lambda value: transform(value)[keep], raw
        )
        expected = torch.linalg.slogdet(jacobian)[1]
        actual = transform.log_abs_det_jacobian(
            raw.unsqueeze(0), transform(raw).unsqueeze(0)
        )[0]
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0)


def test_block_transform_log_det_keeps_the_batch_shape():
    transform = _ces_transform()
    for batch_shape in [(4,), (3, 5), (2, 3, 7)]:
        raw = torch.randn(*batch_shape, 4)
        assert transform(raw).shape == (*batch_shape, 5)
        assert transform.log_abs_det_jacobian(raw, transform(raw)).shape == batch_shape


def test_block_transform_rejects_inconsistent_sizes():
    with pytest.raises(ValueError, match="equal length"):
        BlockTransform([ExpTransform()], raw_sizes=[1, 1], theta_sizes=[1])
    with pytest.raises(ValueError, match="positive"):
        BlockTransform([ExpTransform()], raw_sizes=[0], theta_sizes=[1])


def test_block_transform_rejects_a_wrong_trailing_size():
    transform = _ces_transform()
    with pytest.raises(ValueError, match="trailing size 4"):
        transform(torch.randn(3, 5))
    with pytest.raises(ValueError, match="trailing size 5"):
        transform.inv(torch.randn(3, 4))


def test_ces_posterior_mixture_is_built_over_the_raw_coordinates():
    task = CES()
    net = _make_ces_posterior(task, n_components=3)

    assert net.dim_theta == 5
    assert net.dim_latent == 4
    assert net.loc_head.out_features == 3 * 4
    assert net.log_scale_head.out_features == 3 * 4
    assert net.theta_loc.shape == (4,)


def test_ces_posterior_log_prob_shape_and_scale():
    # The plain mixture reaches ~-4e6 on these same draws, since u exceeds any
    # scale a component can express. This is the regression test for that.
    torch.manual_seed(0)
    task = CES()
    net = _make_ces_posterior(task)
    y_history, design_history = _ces_history(task, b=512)
    theta = task.sample_prior(512)

    log_q = net.log_prob(theta, y_history, design_history)

    assert log_q.shape == (512,)
    assert torch.isfinite(log_q).all()
    assert log_q.min() > -1e3


def test_ces_posterior_rejects_wrong_theta_shape():
    # Must stay a ValueError from the shape check, not an IndexError from
    # slicing inside the transform.
    task = CES()
    net = _make_ces_posterior(task)
    y_history, design_history = _ces_history(task, b=4)

    with pytest.raises(ValueError, match="trailing shape"):
        net.log_prob(task.sample_prior(4)[:, :4], y_history, design_history)


def test_ces_posterior_samples_satisfy_the_constraints():
    torch.manual_seed(0)
    task = CES()
    net = _make_ces_posterior(task)
    y_history, design_history = _ces_history(task, b=8)

    theta = net.sample(y_history, design_history, n_samples=3)

    assert theta.shape == (3, 8, 5)
    assert ((theta[..., 0] > 0.01) & (theta[..., 0] < 1.0)).all()
    assert torch.allclose(
        theta[..., 1:4].sum(-1), torch.ones_like(theta[..., 0]), atol=1e-5
    )
    assert (theta[..., 4] > 0).all()


def test_ces_posterior_has_no_distribution_object_over_theta():
    task = CES()
    net = _make_ces_posterior(task)
    y_history, design_history = _ces_history(task, b=4)

    assert net.raw_distribution(y_history, design_history).log_prob(
        torch.randn(4, 4)
    ).shape == (4,)
    with pytest.raises(NotImplementedError, match="raw_distribution"):
        net.distribution(y_history, design_history)


def test_constrained_posterior_rejects_canonicalize_theta():
    task = LocationFinding(n_sources=2, dim_theta=2)
    with pytest.raises(ValueError, match="canonicalize_theta"):
        ConstrainedAttentiveVariationalPosteriorNet(
            event_shape_y=task.event_shape_y,
            event_shape_design=task.event_shape_design,
            event_shape_theta=(4,),
            transform=BlockTransform([ExpTransform()], raw_sizes=[4], theta_sizes=[4]),
            canonicalize_theta=True,
        )


def test_constrained_posterior_rejects_a_transform_of_the_wrong_width():
    task = CES()
    with pytest.raises(ValueError, match="as many coordinates as theta"):
        ConstrainedAttentiveVariationalPosteriorNet(
            event_shape_y=task.event_shape_y,
            event_shape_design=task.event_shape_design,
            event_shape_theta=task.event_shape_theta,
            transform=BlockTransform([ExpTransform()], raw_sizes=[4], theta_sizes=[4]),
        )


def test_ces_barber_agakov_gain_is_bounded_for_an_untrained_posterior():
    # Direct regression test for the observed training gains of -2e3 to -3.7e6.
    torch.manual_seed(0)
    task = CES()
    objective = BarberAgakov(task, _make_ces_posterior(task))
    y_history, design_history = _ces_history(task, b=512)
    rollout = Rollout(
        theta=task.sample_prior(512),
        y_history=y_history,
        design_history=design_history,
    )

    gain = objective.per_sample_gain(rollout)

    assert torch.isfinite(gain).all()
    assert gain.abs().max() < 1e3


def test_ces_posterior_pushforward_normalises():
    """The measure claim the whole design rests on, made falsifiable.

    Integrates the CES prior over the raw coordinates by importance sampling. It
    comes to 0.99 rather than 1 because CESPrior.log_prob does not account for
    its own `rho = 0.01 + 0.99 * rho` squash, which makes the reported bound
    conservative by log(0.99) = 0.01 nats. Also the drift guard on the recipe's
    0.01 rho floor: change it in one place only and this fails.
    """
    torch.manual_seed(0)
    task = CES()
    transform = _ces_transform()

    raw = transform.inv(task.sample_prior(20_000))
    proposal = Independent(Normal(raw.mean(0), 2.0 * raw.std(0)), 1)
    z = proposal.sample((200_000,))
    log_weights = (
        task.prior.log_prob(transform(z))
        + transform.log_abs_det_jacobian(z, transform(z))
        - proposal.log_prob(z)
    )

    assert log_weights.exp().mean().item() == pytest.approx(0.99, rel=0.05)


def test_michaelis_menten_posterior_pushforward_normalises():
    """The sigmoid-affine map must land on exactly the prior's (20, 200) box.

    Unlike CES there is no squash the prior fails to account for, so this
    integrates to 1. A drift guard on the bounds: change MichaelisMentenPrior's
    range or the recipe's affine alone and this fails.
    """
    torch.manual_seed(0)
    task = MichaelisMenten()
    transform = BlockTransform(
        transforms=[
            ComposeTransform(
                [SigmoidTransform(), AffineTransform(loc=20.0, scale=180.0)]
            )
        ],
        raw_sizes=[2],
        theta_sizes=[2],
    )

    raw = transform.inv(task.sample_prior(20_000))
    proposal = Independent(Normal(raw.mean(0), 2.0 * raw.std(0)), 1)
    z = proposal.sample((200_000,))
    log_weights = (
        task.prior.log_prob(transform(z))
        + transform.log_abs_det_jacobian(z, transform(z))
        - proposal.log_prob(z)
    )

    assert log_weights.exp().mean().item() == pytest.approx(1.0, rel=0.05)


def test_ces_posterior_gradients_reach_the_posterior_parameters():
    torch.manual_seed(0)
    task = CES()
    net = _make_ces_posterior(task)
    y_history, design_history = _ces_history(task, b=32)
    rollout = Rollout(
        theta=task.sample_prior(32),
        y_history=y_history,
        design_history=design_history,
    )

    BarberAgakov(task, net).estimate_gain(rollout).backward()

    trainable = [p for n, p in net.named_parameters() if n != "encoder.null_enc"]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)


def test_log_det_correction_does_not_change_the_posterior_gradient():
    """The log-det is a function of theta alone, which carries no graph.

    So correcting the density shifts the reported bound without touching any
    gradient. That is what makes the correction safe to apply inside the net.
    """
    torch.manual_seed(0)
    task = CES()
    net = _make_ces_posterior(task)
    y_history, design_history = _ces_history(task, b=32)
    theta = task.sample_prior(32)

    def grads(loss):
        net.zero_grad()
        loss.backward()
        return torch.cat(
            [p.grad.flatten() for p in net.parameters() if p.grad is not None]
        )

    corrected = grads(net.log_prob(theta, y_history, design_history).sum())
    raw_only = grads(
        net.raw_distribution(y_history, design_history)
        .log_prob(net.transform.inv(theta))
        .sum()
    )

    torch.testing.assert_close(corrected, raw_only)


def _htd_transform():
    """The HTD map: (z_k, alpha) -> (k, alpha). exp on k, identity on alpha.

    Mirrors the transform in experiments/temporal_discounting/joint_var_post.yaml.
    """
    return BlockTransform(
        transforms=[ExpTransform(), AffineTransform(loc=0.0, scale=1.0)],
        raw_sizes=[1, 1],
        theta_sizes=[1, 1],
    )


def _make_htd_posterior(task, n_components=3):
    return ConstrainedAttentiveVariationalPosteriorNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_theta=task.event_shape_theta,
        transform=_htd_transform(),
        n_components=n_components,
        encoder_dim_h=[16],
        dim_emb=16,
        num_heads=2,
        num_blocks=1,
        trunk_dim_h=16,
    )


def _htd_history(task, b=16, T=3):
    return torch.randint(0, 2, (b, T, 1)).float(), torch.rand(b, T, 2) * 100


def test_htd_transform_round_trips_on_real_prior_draws():
    torch.manual_seed(0)
    transform = _htd_transform()

    raw = torch.randn(64, 2)
    assert torch.allclose(transform.inv(transform(raw)), raw, atol=1e-5)

    theta = TemporalDiscounting().sample_prior(2_000)
    assert torch.allclose(transform(transform.inv(theta)), theta, atol=1e-5)
    assert torch.isfinite(transform.inv(theta)).all()


def test_htd_posterior_pushforward_normalises():
    """The measure claim, for HTD.

    Only k is transformed, so the Jacobian is log k alone. This integrates to
    exactly 1 over the prior's support: the alpha proposal is kept positive
    because HalfNormal.log_prob rejects negatives outright.
    """
    torch.manual_seed(0)
    task = TemporalDiscounting()
    transform = _htd_transform()

    raw = transform.inv(task.sample_prior(20_000))
    z_k = Normal(raw[:, 0].mean(), 2.0 * raw[:, 0].std())
    z_alpha = HalfNormal(4.0)
    z = torch.stack([z_k.sample((200_000,)), z_alpha.sample((200_000,))], dim=-1)
    log_weights = (
        task.prior.log_prob(transform(z))
        + transform.log_abs_det_jacobian(z, transform(z))
        - z_k.log_prob(z[:, 0])
        - z_alpha.log_prob(z[:, 1])
    )

    assert log_weights.exp().mean().item() == pytest.approx(1.0, rel=0.05)


def test_htd_constrained_posterior_keeps_k_positive():
    """k is the transformed coordinate, so only it is positive by construction.

    alpha is modelled in its own coordinates and may leak below zero; that leak
    is the accepted cost of not putting alpha through a log.
    """
    torch.manual_seed(0)
    task = TemporalDiscounting()
    net = _make_htd_posterior(task)
    y_history, design_history = _htd_history(task, b=8)

    theta = net.sample(y_history, design_history, n_samples=64)

    assert theta.shape == (64, 8, 2)
    assert (theta[..., 0] > 0).all()


def test_htd_log_prob_is_tame_for_a_near_zero_alpha():
    """Regression test for the NaN that killed sweep 3tu0ab2h.

    alpha ~ HalfNormal(2) has its mode at zero, so draws near zero are routine.
    Under exp on both coordinates such a draw landed ~13 spreads into a tail no
    Gaussian covers, and the gradient chasing it overflowed. Untransformed it is
    an ordinary point, so log_prob must stay close to that of a central alpha.
    """
    torch.manual_seed(0)
    task = TemporalDiscounting()
    net = _make_htd_posterior(task)
    y_history, design_history = _htd_history(task, b=2)

    k = task.sample_prior(2)[0, 0]
    theta = torch.stack(
        [torch.tensor([k, 2.6e-7]), torch.tensor([k, 1.6])]  # tail alpha, central
    )
    log_q = net.log_prob(theta, y_history, design_history)

    assert torch.isfinite(log_q).all()
    assert (log_q[0] - log_q[1]).abs() < 5.0


def test_htd_barber_agakov_gain_is_finite_for_an_untrained_posterior():
    torch.manual_seed(0)
    task = TemporalDiscounting()
    objective = BarberAgakov(task, _make_htd_posterior(task))
    y_history, design_history = _htd_history(task, b=512)
    rollout = Rollout(
        theta=task.sample_prior(512),
        y_history=y_history,
        design_history=design_history,
    )

    gain = objective.per_sample_gain(rollout)

    assert torch.isfinite(gain).all()
    assert gain.abs().max() < 1e3
