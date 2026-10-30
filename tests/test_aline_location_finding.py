"""Tests for the BED-focused ALINE implementation."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from testbed.core import Box
from testbed.core.hooks import ActionHook, PosteriorUpdateHook
from testbed.core.rollout import Rollout
from testbed.design_nets import AlineModel, AlineOutput
from testbed.designers import AlineDesigner
from testbed.methods.aline import Trainer, load_checkpoint
from testbed.tasks import LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2, design_prior=Box(-4.0, 4.0, (2,)))


def _make_cfg(**overrides):
    values = {
        "T": 3,
        "pool_size": 4,
        "eval_pool_size": 5,
        "eval_batch_size": 2,
        "n_outer": 4,
        "n_outer_eval": 4,
        "n_inner_eval": 2,
        "n_train_iter": 2,
        "n_warmup_iter": 1,
        "eval_every": 0,
        "eval_deterministic": False,
        "gamma": 1.0,
        "alpha": 1.0,
        "grad_clip": 1.0,
        "optimizer": SimpleNamespace(lr=1e-3, weight_decay=0.01),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_model(task):
    return AlineModel(
        event_shape_y=task.event_shape_y,
        event_shape_theta=task.event_shape_theta,
        event_shape_design=task.event_shape_design,
        embedding_dim=8,
        feedforward_dim=16,
        n_heads=2,
        n_layers=1,
        n_components=3,
    )


def _make_trainer(cfg=None, manager=None, task=None):
    task = _make_task() if task is None else task
    model = _make_model(task)
    trainer = Trainer(
        task,
        model,
        cfg or _make_cfg(),
        manager=manager,
    )
    return trainer, model


def test_model_masks_unavailable_candidates_and_scores_posterior():
    task = _make_task()
    space = Box(-4.0, 4.0, event_shape=task.event_shape_design)
    model = _make_model(task)
    context_designs = space.sample((2, 2))
    context_y = task.sample_outcome(
        task.sample_prior(2).unsqueeze(1).expand(-1, 2, -1, -1),
        context_designs,
    )
    candidates = space.sample((2, 5))
    available = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]]
    )

    output = model(
        context_y,
        context_designs,
        candidates,
        available_mask=available,
    )

    policy = torch.distributions.Categorical(logits=output.acquisition_logits)
    assert (policy.probs[~available] == 0).all()
    assert (policy.probs[available] > 0).all()
    assert torch.isfinite(
        output.posterior.log_prob(task.sample_prior(2).reshape(2, 2))
    ).all()


def test_fractional_outcome_scale_is_not_truncated_by_integer_center():
    task = _make_task()
    model = AlineModel(
        event_shape_y=task.event_shape_y,
        event_shape_theta=task.event_shape_theta,
        event_shape_design=task.event_shape_design,
        embedding_dim=8,
        feedforward_dim=16,
        n_heads=2,
        n_layers=1,
        n_components=3,
        outcome_center=0,
        outcome_scale=0.5,
    )

    torch.testing.assert_close(
        model._normalize_outcomes(torch.tensor([[[1.0]]])),
        torch.tensor([[[2.0]]]),
    )


def test_fractional_design_scale_is_not_truncated_by_integer_center():
    task = _make_task()
    model = AlineModel(
        event_shape_y=task.event_shape_y,
        event_shape_theta=task.event_shape_theta,
        event_shape_design=task.event_shape_design,
        embedding_dim=8,
        feedforward_dim=16,
        n_heads=2,
        n_layers=1,
        n_components=3,
        design_center=0,
        design_scale=0.5,
    )

    torch.testing.assert_close(
        model._normalize_designs(torch.tensor([[[1.0, 1.0]]])),
        torch.tensor([[[2.0, 2.0]]]),
    )


def test_attention_prevents_candidate_leakage():
    torch.manual_seed(0)
    task = _make_task()
    space = Box(-4.0, 4.0, event_shape=task.event_shape_design)
    model = _make_model(task).eval()
    theta = task.sample_prior(2)[:1]
    context_designs = space.sample((1, 2))
    context_y = task.sample_outcome(
        theta.unsqueeze(1).expand(-1, 2, -1, -1), context_designs
    )
    candidates = space.sample((1, 3))
    changed_candidates = candidates.clone()
    changed_candidates[:, 1:] = -changed_candidates[:, 1:]

    output = model(context_y, context_designs, candidates)
    changed = model(context_y, context_designs, changed_candidates)

    torch.testing.assert_close(
        output.acquisition_logits[:, 0], changed.acquisition_logits[:, 0]
    )
    torch.testing.assert_close(output.mixture_logits, changed.mixture_logits)
    torch.testing.assert_close(output.mixture_means, changed.mixture_means)
    torch.testing.assert_close(output.mixture_scales, changed.mixture_scales)


def test_rollout_samples_in_bounds_without_replacement():
    torch.manual_seed(1)
    trainer, _ = _make_trainer()
    theta = trainer.task.sample_prior(4)

    rollout = trainer.designer.rollout(
        theta, deterministic=True, return_posterior_scores=True
    )

    assert trainer.task.contains_design(rollout.design_history)
    for designs in rollout.design_history:
        assert torch.unique(designs, dim=0).shape[0] == trainer.cfg.T


@pytest.mark.parametrize("random_policy", [False, True])
def test_shared_rollout_keeps_one_pool_and_masks_selected_candidates(random_policy):
    trainer, model = _make_trainer()
    designer = trainer.designer
    seen = []
    handle = model.register_forward_pre_hook(
        lambda module, inputs, kwargs: seen.append(
            (inputs[2], kwargs["available_mask"].clone())
        ),
        with_kwargs=True,
    )
    theta = trainer.task.sample_prior(2).requires_grad_()
    rollout = designer.rollout(
        theta,
        random_policy=random_policy,
        pool_size=designer.T,
        return_posterior_scores=True,
    )
    handle.remove()

    assert len(seen) == designer.T + 1
    candidates = seen[0][0]
    for t, (pool, mask) in enumerate(seen):
        torch.testing.assert_close(pool, candidates)
        assert torch.all(mask.sum(dim=1) == designer.T - t)
        for b in range(2):
            for design in rollout.design_history[b, :t]:
                selected = (pool[b] == design).all(dim=-1)
                assert selected.sum() == 1
                assert not mask[b, selected].any()
    assert not rollout.y_history.requires_grad
    assert rollout.extra["posterior_log_probs"].requires_grad


def test_posterior_collection_preserves_deterministic_evaluation_trajectory():
    trainer, model = _make_trainer()
    model.eval()
    theta = trainer.task.sample_prior(2)
    torch.manual_seed(12)
    ordinary = trainer.designer.rollout(theta, deterministic=True)
    torch.manual_seed(12)
    scored = trainer.designer.rollout(
        theta, deterministic=True, return_posterior_scores=True
    )
    torch.testing.assert_close(ordinary.design_history, scored.design_history)
    torch.testing.assert_close(ordinary.y_history, scored.y_history)
    torch.testing.assert_close(
        ordinary.extra["policy_log_probs"], scored.extra["policy_log_probs"]
    )


@pytest.mark.parametrize("n_sources, dim_theta", [(1, 2), (2, 3)])
def test_mean_log_prob_includes_every_parameter(n_sources, dim_theta):
    task = LocationFinding(
        n_sources=n_sources,
        dim_theta=dim_theta,
        design_prior=Box(-4.0, 4.0, (dim_theta,)),
    )
    model = _make_model(task)
    trainer = Trainer(task, model, _make_cfg())
    n_parameters = n_sources * dim_theta
    output = AlineOutput(
        acquisition_logits=torch.zeros(2, 1),
        mixture_logits=torch.zeros(2, n_parameters, 1),
        mixture_means=torch.zeros(2, n_parameters, 1),
        mixture_scales=torch.ones(2, n_parameters, 1),
    )
    theta = torch.ones(2, n_sources, dim_theta, requires_grad=True)
    actual = trainer.designer._mean_log_prob(output, theta)
    expected = torch.distributions.Normal(0, 1).log_prob(theta).mean(dim=(1, 2))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(theta.grad, torch.full_like(theta, -1 / n_parameters))


def test_single_outer_sample_has_finite_losses_and_gradients():
    torch.manual_seed(4)
    trainer, model = _make_trainer(cfg=_make_cfg(n_outer=1, n_warmup_iter=0))
    metrics = trainer.train_step(step=0)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_policy_reward_matches_reference_clipping_and_standardization():
    trainer, _ = _make_trainer(cfg=_make_cfg(n_outer=2, n_warmup_iter=0, gamma=0.5))
    rollout = Rollout(
        theta=trainer.task.sample_prior(2),
        y_history=torch.empty(2, 0, 1),
        design_history=torch.empty(2, 0, 2),
        extra={
            "posterior_log_probs": torch.tensor(
                [[0.0, -2.0, 1.0], [0.0, -1.0, -2.0]], requires_grad=True
            ),
            "policy_log_probs": torch.tensor(
                [[1.0, 2.0], [4.0, 4.0]], requires_grad=True
            ),
        },
    )
    trainer.designer.rollout = lambda theta, **kwargs: rollout

    metrics = trainer.train_step(step=0)

    assert metrics["policy_loss"] == pytest.approx(1 / (2 * 2**0.5))
    torch.testing.assert_close(
        rollout.extra["posterior_log_probs"].grad,
        torch.full_like(rollout.extra["posterior_log_probs"], -1 / 6),
    )


@pytest.mark.parametrize("training", [True, False])
def test_empty_context_has_finite_outputs_and_gradients(training):
    trainer, model = _make_trainer()
    model.train(training)
    candidates = trainer.task.design_prior.sample((2, 3))
    output = model(
        torch.empty(2, 0, 1),
        torch.empty(2, 0, 2),
        candidates,
    )
    loss = output.mixture_means.sum() + output.acquisition_logits.sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_rollout_passes_history_and_trains_every_posterior_state():
    trainer, model = _make_trainer()
    sample_outcome = trainer.task.sample_outcome
    lengths = []

    def sample(theta, design, y_history=None, design_history=None):
        lengths.append(y_history.shape[1])
        assert y_history.shape[1] == design_history.shape[1]
        return sample_outcome(
            theta, design, y_history=y_history, design_history=design_history
        )

    trainer.task.sample_outcome = sample
    outputs = []
    handle = model.register_forward_hook(
        lambda module, inputs, output: outputs.append(output)
    )
    theta = trainer.task.sample_prior(4)
    rollout = trainer.designer.rollout(theta, return_posterior_scores=True)
    handle.remove()
    assert lengths == [0, 1, 2]
    expected = torch.stack(
        [out.posterior.log_prob(theta.reshape(4, 2)).mean(-1) for out in outputs],
        dim=1,
    )
    torch.testing.assert_close(rollout.extra["posterior_log_probs"], expected)


def _checkpoint_cfg(**overrides):
    values = vars(_make_cfg()).copy()
    values["optimizer"] = vars(values["optimizer"])
    values.update(
        device="cpu",
        train_time_budget_seconds=None,
        checkpoint_every=1,
        n_train_iter=4,
        task={
            "_target_": "testbed.tasks.LocationFinding",
            "n_sources": 1,
            "dim_theta": 2,
            "design_prior": {
                "_target_": "testbed.core.Box",
                "low": -4.0,
                "high": 4.0,
                "event_shape": [2],
            },
        },
        design_net={
            "_target_": "testbed.design_nets.AlineModel",
            "embedding_dim": 8,
            "feedforward_dim": 16,
            "n_heads": 2,
            "n_layers": 1,
            "n_components": 3,
        },
    )
    values.update(overrides)
    return OmegaConf.create(values)


def _checkpoint_manager(tmp_path):
    return SimpleNamespace(
        path=lambda name: tmp_path / name,
        event_logger=SimpleNamespace(info=lambda message: None),
        metric_logger=None,
    )


@pytest.mark.parametrize("stop_after", [1, 2])
def test_checkpoint_resumes_exactly_across_optimizer_phases(
    tmp_path, monkeypatch, stop_after
):
    import testbed.methods.aline as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    module.seed_all(13)
    baseline, _ = _make_trainer(_checkpoint_cfg())
    baseline.train()
    expected_rng = module.rng_state()

    module.seed_all(13)
    trainer, _ = _make_trainer(
        _checkpoint_cfg(train_time_budget_seconds=10),
        manager=_checkpoint_manager(tmp_path),
    )
    original_step = trainer.train_step

    def timed_step(step):
        result = original_step(step)
        if step + 1 == stop_after:
            clock[0] = 12.0
        return result

    monkeypatch.setattr(trainer, "train_step", timed_step)
    trainer.train()
    saved = torch.load(tmp_path / "final_step.pt", weights_only=True)
    assert saved["stop_reason"] == "time_budget"
    assert saved["completed_iterations"] == stop_after
    assert saved["step"] == stop_after - 1
    assert saved["elapsed_seconds"] == 12.0
    assert saved["joint_optimizer_started"] == (stop_after > 1)
    policy = torch.load(tmp_path / f"step_{stop_after:08d}.pt", weights_only=True)
    assert "optimizer" not in policy
    torch.testing.assert_close(policy["design_net"], saved["design_net"])
    with pytest.raises(ValueError, match="time budget is exhausted"):
        load_checkpoint(tmp_path / "final_step.pt")
    with pytest.raises(ValueError, match="policy-only"):
        load_checkpoint(tmp_path / f"step_{stop_after:08d}.pt")

    # Downtime is excluded, while the already-used 12 seconds are retained.
    clock[0] = 100.0
    resumed = load_checkpoint(
        tmp_path / "final_step.pt", train_time_budget_seconds=float("inf")
    )
    assert resumed.best_metric == trainer.best_metric
    assert resumed._start_time == 88.0
    resumed.train()
    assert resumed.completed_iterations == 4
    torch.testing.assert_close(
        resumed.model.state_dict(), baseline.model.state_dict(), rtol=0, atol=0
    )
    assert resumed.scheduler.state_dict() == baseline.scheduler.state_dict()
    torch.testing.assert_close(
        resumed.optimizer.state_dict(), baseline.optimizer.state_dict(), rtol=0, atol=0
    )
    actual_rng = module.rng_state()
    assert actual_rng["python"] == expected_rng["python"]
    assert actual_rng["numpy"] == expected_rng["numpy"]
    torch.testing.assert_close(actual_rng["torch"], expected_rng["torch"])


@pytest.mark.parametrize("expired_before_start", [False, True])
def test_deadline_skips_evaluation_and_saves_full_state(
    tmp_path, monkeypatch, expired_before_start
):
    import testbed.methods.aline as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    trainer, _ = _make_trainer(
        _checkpoint_cfg(train_time_budget_seconds=10, eval_every=1, checkpoint_every=0),
        manager=_checkpoint_manager(tmp_path),
    )

    def step(index):
        clock[0] = 11.0
        return {"inference_loss": 1.0}

    monkeypatch.setattr(trainer, "train_step", step)
    monkeypatch.setattr(
        trainer, "evaluate", lambda: pytest.fail("unexpected evaluation")
    )
    if expired_before_start:
        clock[0] = 11.0
    trainer.train()
    state = torch.load(tmp_path / "final_step.pt", weights_only=True)
    assert state["completed_iterations"] == int(not expired_before_start)
    assert state["step"] == (-1 if expired_before_start else 0)
    assert state["stop_reason"] == "time_budget"
    assert not list(tmp_path.glob("step_*.pt"))


@pytest.mark.parametrize("budget", [None, float("inf")])
def test_unlimited_training_and_iteration_limit_override(tmp_path, budget):
    trainer, _ = _make_trainer(
        _checkpoint_cfg(train_time_budget_seconds=budget),
        manager=_checkpoint_manager(tmp_path),
    )
    trainer.train()
    assert trainer.completed_iterations == 4
    assert trainer.stop_reason == "iteration_limit"
    with pytest.raises(ValueError, match="iteration limit is exhausted"):
        load_checkpoint(tmp_path / "final_step.pt")
    resumed = load_checkpoint(tmp_path / "final_step.pt", n_train_iter=5)
    assert resumed.scheduler.state_dict() == trainer.scheduler.state_dict()
    resumed.train()
    assert resumed.completed_iterations == 5


@pytest.mark.parametrize("budget", [0, -1, float("nan"), True, "10"])
def test_invalid_time_budget(budget):
    with pytest.raises(ValueError, match="train_time_budget_seconds"):
        _make_trainer(_make_cfg(train_time_budget_seconds=budget))


@pytest.mark.parametrize("interval", [-1, 1.5, True, None])
def test_invalid_checkpoint_interval(interval):
    with pytest.raises(ValueError, match="checkpoint_every"):
        _make_trainer(_make_cfg(checkpoint_every=interval))


class _SumActionNet(torch.nn.Module):
    """Stands in for an action net: one number per rollout from the history."""

    def forward(self, y_history, design_history):
        return y_history.sum(dim=(1, 2))


def _make_aline_designer(task):
    return AlineDesigner(task, T=3, model=_make_model(task), pool_size=4)


def test_rollout_runs_a_terminal_action_hook():
    task = _make_task()
    designer = _make_aline_designer(task)
    designer.hooks.append(ActionHook(_SumActionNet()))

    rollout = designer.rollout(task.sample_prior(2))

    assert rollout.extra["action"].shape == (2,)


@pytest.mark.parametrize(
    "hook",
    [PosteriorUpdateHook(), ActionHook(_SumActionNet(), every_step=True)],
)
def test_rollout_rejects_hooks_it_cannot_run(hook):
    task = _make_task()
    designer = _make_aline_designer(task)
    designer.hooks.append(hook)

    with pytest.raises(NotImplementedError, match="per-step callbacks"):
        designer.rollout(task.sample_prior(2))
