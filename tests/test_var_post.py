"""Tests for the variational-posterior (Barber-Agakov) EIG estimator."""

import pytest
import torch
from torch.distributions import Independent, MultivariateNormal, Normal

from testbed.core import RandomScaleGaussianDesignDistribution
from testbed.core.design_spaces import BoxDesignSpace
from testbed.core.hooks import RolloutHook
from testbed.core.util import sort_by_first_coordinate
from testbed.design_nets import (
    AttentiveGuideNet,
    DeepSetGuideNet,
    DeepSetPolicyNet,
    TanhGaussianPolicyNet,
)
from testbed.design_nets.guide import GaussianMixtureGuideNet, GuideNet, GuideNLL
from testbed.design_objectives import VarPost, VarPostReinforce
from testbed.design_objectives.reinforce import leave_one_out_baseline
from testbed.designers import RandomDesigner, Rollout, build_policy_designer
from testbed.tasks import (
    HyperbolicTemporalDiscounting,
    LinearRegression,
    LocationFinding,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_task():
    return HyperbolicTemporalDiscounting()


def _make_design_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _make_stochastic_net(task):
    space = BoxDesignSpace(low=[5.0, 20.0], high=[40.0, 80.0], event_shape=(2,))
    return TanhGaussianPolicyNet(space)


def _make_guide(task, n_components=3):
    return DeepSetGuideNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=n_components,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
    )


def _make_rollout(task, policy_net, b=8, T=3, **kwargs):
    designer = build_policy_designer(task, T=T, design_net=policy_net, **kwargs)
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


class _ExactPosteriorGuide(GuideNet):
    """Test-only guide returning the exact Gaussian-linear posterior.

    Only valid for LinearRegression(dim=1) with a single (T=1) observation and
    a fixed design; used to check VarPost's bound is tight (matches the exact
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


# ============================================================================
# GuideNet: shapes and mixture correctness
# ============================================================================


@pytest.mark.parametrize(
    "net_fn",
    [
        lambda task: DeepSetGuideNet(
            task.event_shape_y,
            task.event_shape_design,
            task.event_shape_theta,
            n_components=3,
            encoder_dim_h=[16],
            dim_emb=16,
            trunk_dim_h=16,
        ),
        lambda task: AttentiveGuideNet(
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
def test_guide_log_prob_shape_and_finite_1d_theta(net_fn):
    task = _make_task()  # HTD: event_shape_theta == [2]
    guide = net_fn(task)
    rollout = _make_rollout(task, _make_design_net(task))

    log_q = guide.log_prob(rollout.theta, rollout.y_history, rollout.design_history)
    assert log_q.shape == (rollout.b,)
    assert torch.isfinite(log_q).all()


def test_guide_log_prob_multidim_theta_flattens_correctly():
    task = LocationFinding(n_sources=2, dim_theta=2)  # event_shape_theta == [2, 2]
    guide = _make_guide(task)
    designer = RandomDesigner(task, T=3)
    theta0 = task.sample_prior(6)
    rollout = designer.rollout(theta0)

    log_q = guide.log_prob(rollout.theta, rollout.y_history, rollout.design_history)
    assert log_q.shape == (6,)
    assert torch.isfinite(log_q).all()


def test_guide_log_prob_rejects_wrong_theta_shape():
    task = _make_task()
    guide = _make_guide(task)
    rollout = _make_rollout(task, _make_design_net(task))
    with pytest.raises(ValueError):
        guide.log_prob(rollout.theta[:, :1], rollout.y_history, rollout.design_history)


def test_guide_sample_shapes():
    task = _make_task()
    guide = _make_guide(task)
    rollout = _make_rollout(task, _make_design_net(task), b=5)

    s1 = guide.sample(rollout.y_history, rollout.design_history, n_samples=1)
    s3 = guide.sample(rollout.y_history, rollout.design_history, n_samples=3)
    assert s1.shape == (
        1,
        5,
        *task.event_shape_theta,
    )  # leading n_samples dim always kept
    assert s3.shape == (3, 5, *task.event_shape_theta)


def test_mixture_with_one_component_matches_independent_normal():
    torch.manual_seed(0)
    task = _make_task()
    guide = DeepSetGuideNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=1,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
    )
    rollout = _make_rollout(task, _make_design_net(task), b=4)

    dist = guide.distribution(rollout.y_history, rollout.design_history)
    loc = dist.component_distribution.base_dist.loc[:, 0]
    scale = dist.component_distribution.base_dist.scale[:, 0]
    reference = Independent(Normal(loc, scale), 1)

    theta_flat = rollout.theta.reshape(rollout.b, guide.dim_theta)
    torch.testing.assert_close(
        dist.log_prob(theta_flat), reference.log_prob(theta_flat)
    )


def test_mixture_with_three_components_matches_manual_logsumexp():
    torch.manual_seed(1)
    task = _make_task()
    guide = _make_guide(task, n_components=3)
    rollout = _make_rollout(task, _make_design_net(task), b=4)

    dist = guide.distribution(rollout.y_history, rollout.design_history)
    theta_flat = rollout.theta.reshape(rollout.b, guide.dim_theta)

    loc = dist.component_distribution.base_dist.loc  # [b, K, D]
    scale = dist.component_distribution.base_dist.scale
    logits = dist.mixture_distribution.logits  # [b, K]

    comp_logp = Normal(loc, scale).log_prob(theta_flat.unsqueeze(1)).sum(-1)  # [b, K]
    manual = torch.logsumexp(torch.log_softmax(logits, dim=-1) + comp_logp, dim=-1)

    torch.testing.assert_close(dist.log_prob(theta_flat), manual)


# ============================================================================
# VarPost: bound correctness against a closed-form EIG
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


def test_untrained_guide_gives_a_bound_strictly_below_exact_eig():
    torch.manual_seed(0)
    x_val = 0.7
    task, *_rest, exact_eig = _linear_regression_setup(x_val=x_val)
    rollout = _fixed_design_rollout(task, x_val, b=8192)

    guide = DeepSetGuideNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=3,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
    )
    bound = VarPost(task, guide).estimate_gain(rollout)

    assert (
        bound.item() < exact_eig.item() - 0.5
    )  # comfortably below, well outside MC noise


def test_exact_posterior_guide_bound_matches_exact_eig():
    torch.manual_seed(1)
    x_val = 0.7
    task, prior_loc, prior_cov, design_row, exact_eig = _linear_regression_setup(
        x_val=x_val
    )
    rollout = _fixed_design_rollout(task, x_val, b=32768)

    guide = _ExactPosteriorGuide(prior_loc, prior_cov, design_row, task.noise_scale)
    bound = VarPost(task, guide).estimate_gain(rollout)

    assert bound.item() == pytest.approx(exact_eig.item(), abs=0.05)


# ============================================================================
# VarPost: gradient flow and freezing
# ============================================================================


def test_var_post_gradient_reaches_guide_params():
    task = _make_task()
    guide = _make_guide(task)
    rollout = _make_rollout(task, _make_design_net(task))

    gain = VarPost(task, guide).estimate_gain(rollout)
    gain.backward()

    trainable = [p for n, p in guide.named_parameters() if n != "encoder.null_enc"]
    assert all(p.grad is not None for p in trainable)
    assert any(p.grad.abs().sum() > 0 for p in trainable)


def test_var_post_gradient_reaches_deterministic_policy_and_guide_in_one_backward():
    task = _make_task()
    guide = _make_guide(task)
    policy_net = _make_design_net(task)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(task.sample_prior(8))

    gain = VarPostReinforce(task, guide, stochastic_design_policy=False).estimate_gain(
        rollout, train_guide=True
    )
    gain.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy_net.parameters()
    )
    trainable = [p for n, p in guide.named_parameters() if n != "encoder.null_enc"]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)


def test_var_post_reinforce_train_guide_false_blocks_guide_gradient_even_if_unfrozen():
    """train_guide=False must guarantee zero guide gradient in the
    deterministic/pathwise branch too, regardless of the guide's own
    requires_grad -- not merely rely on the caller having frozen it."""
    task = _make_task()
    guide = _make_guide(task)
    policy_net = _make_design_net(task)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(task.sample_prior(8))
    objective = VarPostReinforce(task, guide, stochastic_design_policy=False)

    gain = objective.estimate_gain(rollout, train_guide=False)
    gain.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy_net.parameters()
    )
    trainable = [p for n, p in guide.named_parameters() if n != "encoder.null_enc"]
    assert all(p.grad is None for p in trainable)
    # requires_grad is restored afterward, not permanently disabled
    assert all(p.requires_grad for p in guide.parameters())


def test_frozen_guide_receives_no_gradient():
    task = _make_task()
    guide = _make_guide(task)
    guide.eval()
    for p in guide.parameters():
        p.requires_grad_(False)

    rollout = _make_rollout(task, _make_design_net(task))
    gain = VarPost(task, guide).estimate_gain(rollout)
    gain.backward()

    assert all(p.grad is None for p in guide.parameters())


def test_var_post_raises_on_non_evaluable_prior():
    task = _make_task()
    guide = _make_guide(task)
    task.prior.log_prob = lambda *_: (_ for _ in ()).throw(NotImplementedError())
    with pytest.raises(ValueError):
        VarPost(task, guide)


# ============================================================================
# VarPostReinforce: matches VarPost forward value, and DAD-Eq.16-style gradient
# ============================================================================


def test_var_post_reinforce_forward_value_matches_var_post():
    task = _make_task()
    guide = _make_guide(task)
    rollout = _make_rollout(task, _make_design_net(task))

    torch.manual_seed(2)
    reinforce_gain = VarPostReinforce(
        task, guide, stochastic_design_policy=False
    ).estimate_gain(rollout)
    torch.manual_seed(2)
    plain_gain = VarPost(task, guide).estimate_gain(rollout)

    assert torch.allclose(reinforce_gain, plain_gain)


@pytest.mark.parametrize("baseline", [None, "leave_one_out"])
def test_deterministic_reinforce_matches_dad_eq16_reference(baseline):
    torch.manual_seed(0)
    task = _make_task()
    guide = _make_guide(task)
    policy_net = _make_design_net(task)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(task.sample_prior(8))

    var_post = VarPost(task, guide)
    reinforce = VarPostReinforce(
        task, guide, stochastic_design_policy=False, baseline=baseline
    )

    torch.manual_seed(1)
    gain = reinforce.estimate_gain(rollout)

    torch.manual_seed(1)
    reward = var_post.per_sample_gain(rollout)
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
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(task.sample_prior(8))

    guide = _make_guide(task)

    class _ConstantVarPost(VarPost):
        def per_sample_gain(self, rollout):
            return rollout.y_history.new_full((rollout.b,), 3.0)

    reinforce = VarPostReinforce(
        task, guide, stochastic_design_policy=True, baseline="leave_one_out"
    )
    reinforce.var_post = _ConstantVarPost(task, guide)

    gain = reinforce.estimate_gain(rollout)
    grad = _policy_grad(policy_net, gain)

    assert gain.item() == pytest.approx(3.0)
    assert torch.equal(grad, torch.zeros_like(grad))


# ============================================================================
# Joint-training gradient split (mirrors methods/var_post.Trainer.train_step)
# ============================================================================


def test_estimate_gain_train_guide_updates_both_nets_in_one_backward_stochastic():
    """A single estimate_gain(train_guide=True) + one backward() must update
    both design_net (via the REINFORCE surrogate) and guide_net (via the
    injected term) for a stochastic, non-reparameterized policy."""
    task = _make_task()
    guide = _make_guide(task)
    policy_net = _make_stochastic_net(task)
    designer = build_policy_designer(
        task, T=3, design_net=policy_net, reparameterize_policy=False
    )
    rollout = designer.rollout(task.sample_prior(8))
    assert "policy_log_probs" in rollout.extra

    objective = VarPostReinforce(task, guide, stochastic_design_policy=True)

    gain = objective.estimate_gain(rollout, train_guide=True)
    gain.backward()

    trainable_guide = [
        p for n, p in guide.named_parameters() if n != "encoder.null_enc"
    ]
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy_net.parameters()
    )
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable_guide)


def test_train_guide_preserves_the_interpretable_gain_value():
    """The injected guide term must be zero-forward-value: estimate_gain's
    returned number (used for logging / best-checkpoint comparisons) should
    be unaffected by train_guide, only its gradient."""
    task = _make_task()
    guide = _make_guide(task)
    policy_net = _make_stochastic_net(task)
    designer = build_policy_designer(
        task, T=3, design_net=policy_net, reparameterize_policy=False
    )
    rollout = designer.rollout(task.sample_prior(8))
    objective = VarPostReinforce(task, guide, stochastic_design_policy=True)

    plain = objective.estimate_gain(rollout, train_guide=False)
    with_guide = objective.estimate_gain(rollout, train_guide=True)
    torch.testing.assert_close(plain, with_guide)


def test_train_guide_does_not_affect_policy_gradient_deterministic():
    """train_guide only gates guide_net's gradient; design_net's gradient
    must be identical either way in the deterministic/pathwise branch."""
    task = _make_task()
    guide = _make_guide(task)
    policy_net = _make_design_net(task)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(task.sample_prior(8))
    objective = VarPostReinforce(task, guide, stochastic_design_policy=False)

    plain = objective.estimate_gain(rollout, train_guide=False)
    with_guide = objective.estimate_gain(rollout, train_guide=True)
    torch.testing.assert_close(plain, with_guide)

    grad_plain = _policy_grad(policy_net, plain)
    grad_with_guide = _policy_grad(policy_net, with_guide)
    torch.testing.assert_close(grad_plain, grad_with_guide)


def test_guide_gain_method_matches_var_post_estimate_gain():
    task = _make_task()
    guide = _make_guide(task)
    rollout = _make_rollout(task, _make_design_net(task))
    objective = VarPostReinforce(task, guide, stochastic_design_policy=False)

    torch.testing.assert_close(
        objective.guide_gain(rollout), objective.var_post.estimate_gain(rollout)
    )


# ============================================================================
# Trainer smoke tests
# ============================================================================


def _make_trainer(joint):
    from types import SimpleNamespace

    from testbed.methods.var_post import Trainer

    task = _make_task()
    guide = _make_guide(task)
    design_net = _make_design_net(task)
    designer = build_policy_designer(task, T=3, design_net=design_net)
    pretrain_designer = RandomDesigner(task, T=3)
    objective = VarPostReinforce(task, guide, stochastic_design_policy=False)
    guide_loss = GuideNLL(guide)

    cfg = SimpleNamespace(
        joint=joint,
        n_guide_outer=8,
        n_outer=8,
        guide_grad_clip=1.0,
        grad_clip=1.0,
        objective=SimpleNamespace(requires_policy_log_probs=False),
    )
    trainer = Trainer(
        task=task,
        designer=designer,
        guide_net=guide,
        objective=objective,
        guide_loss=guide_loss,
        policy_optimizer=torch.optim.Adam(design_net.parameters(), lr=1e-3),
        guide_optimizer=torch.optim.Adam(guide.parameters(), lr=1e-3),
        cfg=cfg,
        pretrain_designer=pretrain_designer,
    )
    return trainer, design_net, guide


def _snapshot(net):
    return [p.detach().clone() for p in net.parameters()]


def _any_changed(before, net):
    return any(not torch.equal(b, p) for b, p in zip(before, net.parameters()))


def test_guide_pretrain_step_changes_only_guide():
    trainer, design_net, guide = _make_trainer(joint=False)
    design_before, guide_before = _snapshot(design_net), _snapshot(guide)

    metrics = trainer.guide_pretrain_step()

    assert torch.isfinite(torch.tensor(metrics["guide_pretrain/loss"]))
    assert not _any_changed(design_before, design_net)
    assert _any_changed(guide_before, guide)


def test_frozen_train_step_changes_only_design_net():
    trainer, design_net, guide = _make_trainer(joint=False)
    trainer._freeze_guide()
    design_before, guide_before = _snapshot(design_net), _snapshot(guide)

    metrics = trainer.train_step()

    assert torch.isfinite(torch.tensor(metrics["gain"]))
    assert _any_changed(design_before, design_net)
    assert not _any_changed(guide_before, guide)


def test_joint_train_step_changes_both_nets():
    trainer, design_net, guide = _make_trainer(joint=True)
    design_before, guide_before = _snapshot(design_net), _snapshot(guide)

    metrics = trainer.train_step()

    assert torch.isfinite(torch.tensor(metrics["gain"]))
    assert _any_changed(design_before, design_net)
    assert _any_changed(guide_before, guide)


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


def test_canonicalize_theta_makes_var_post_invariant_to_source_order():
    task = LocationFinding(n_sources=2, dim_theta=2)
    guide = DeepSetGuideNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=3,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
    )
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

    var_post_plain = VarPost(task, guide)
    gain = var_post_plain.per_sample_gain(rollout)
    gain_swapped = var_post_plain.per_sample_gain(swapped_rollout)
    assert not torch.allclose(
        gain, gain_swapped
    )  # generically different without canonicalization

    var_post_canon = VarPost(task, guide, canonicalize_theta=True)
    gain_canon = var_post_canon.per_sample_gain(rollout)
    gain_canon_swapped = var_post_canon.per_sample_gain(swapped_rollout)
    torch.testing.assert_close(
        gain_canon, gain_canon_swapped
    )  # identical with canonicalization


def test_guide_nll_canonicalize_theta_matches_var_post_convention():
    task = LocationFinding(n_sources=2, dim_theta=2)
    guide = DeepSetGuideNet(
        task.event_shape_y,
        task.event_shape_design,
        task.event_shape_theta,
        n_components=3,
        encoder_dim_h=[16],
        dim_emb=16,
        trunk_dim_h=16,
    )
    designer = RandomDesigner(task, T=3)
    rollout = designer.rollout(task.sample_prior(6))

    loss = GuideNLL(guide, canonicalize_theta=True, reduction="none")
    manual = -guide.log_prob(
        sort_by_first_coordinate(rollout.theta),
        rollout.y_history,
        rollout.design_history,
    )
    torch.testing.assert_close(loss.from_rollout(rollout), manual)
