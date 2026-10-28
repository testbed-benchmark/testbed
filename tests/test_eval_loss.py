"""Tests for scoring a designer/action-net pair by its losses."""

import pytest
import torch
from omegaconf import OmegaConf

from testbed.design_nets import DeepSetPolicyNet
from testbed.design_objectives import (
    LogLoss,
    LogMSELoss,
    MSELoss,
    PermutationInvariantLogMSELoss,
    PermutationInvariantMSELoss,
    SortedMSELoss,
)
from testbed.designers import RandomDesigner
from testbed.eval.eval_loss import (
    build_losses,
    eval_losses,
    evaluate_loss_checkpoint,
    per_rollout,
)
from testbed.tasks import LocationFinding

T = 3
LOSSES = OmegaConf.create(
    {
        "mse": {"_target_": "testbed.design_objectives.MSELoss"},
        "sorted_mse": {"_target_": "testbed.design_objectives.SortedMSELoss"},
    }
)


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_action_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_theta,
    )


def test_build_losses_scores_one_rollout_at_a_time():
    losses = build_losses(LOSSES, _make_task())

    assert set(losses) == {"mse", "sorted_mse"}
    assert all(loss.reduction == "none" for loss in losses.values())


@pytest.mark.parametrize(
    "cls",
    [
        MSELoss,
        SortedMSELoss,
        PermutationInvariantMSELoss,
        LogMSELoss,
        PermutationInvariantLogMSELoss,
    ],
)
def test_per_rollout_matches_the_losss_own_mean_reduction(cls):
    # The permutation-invariant losses reduce to [b] themselves, as a total
    # over the event, while their "mean" normalizes by it; both must land on
    # the same scale as the loss reported during training.
    torch.manual_seed(0)
    action, theta = torch.randn(64, 2, 2), torch.randn(64, 2, 2)

    values = per_rollout(cls(reduction="none"), action, theta)

    assert values.shape == (64,)
    assert values.mean() == pytest.approx(
        cls(reduction="mean")(action, theta).item(), rel=1e-5
    )


def test_per_rollout_averages_log_loss_over_its_outer_samples():
    task = _make_task()
    theta = task.sample_prior(64)
    action = theta + 0.1 * torch.randn_like(theta)

    torch.manual_seed(0)
    values = per_rollout(LogLoss(task, 8, reduction="none"), action, theta)
    torch.manual_seed(0)
    expected = LogLoss(task, 8, reduction="mean")(action, theta)

    assert values.shape == (64,)
    assert values.mean() == pytest.approx(expected.item(), rel=1e-5)


def test_per_rollout_rejects_a_loss_that_reduced_over_the_batch():
    with pytest.raises(ValueError, match="reduction='none'"):
        per_rollout(MSELoss(), torch.zeros(4, 2), torch.ones(4, 2))


def test_eval_losses_scores_every_loss_on_shared_rollouts():
    task = _make_task()
    designer = RandomDesigner(task, T)
    action_net = _make_action_net(task)

    out = eval_losses(
        task,
        designer,
        action_net,
        build_losses(LOSSES, task),
        n_outer=8,
        outer_batch_size=4,
        verbose=False,
    )

    assert out["n_outer"] == 8
    assert set(out) == {"mse", "mse_se", "sorted_mse", "sorted_mse_se", "n_outer"}
    assert all(out[k] > 0 for k in ("mse", "mse_se", "sorted_mse", "sorted_mse_se"))
    # Scoring borrows the net and the designer, and hands both back as found,
    # so a second call against the same designer works.
    assert action_net.training
    assert len(designer.hooks) == 0
    action_net.eval()
    eval_losses(
        task,
        designer,
        action_net,
        build_losses(LOSSES, task),
        n_outer=4,
        outer_batch_size=4,
        verbose=False,
    )
    assert not action_net.training


def test_eval_losses_scores_in_eval_mode_and_restores_each_modules_mode():
    task = _make_task()
    designer = RandomDesigner(task, T)
    action_net = _make_action_net(task)
    # As freeze_designer leaves it: top level training, children in eval.
    designer.task.eval()
    seen = []

    def spy(action, theta):
        seen.append((designer.training, action_net.training))
        return (action - theta) ** 2

    eval_losses(task, designer, action_net, {"spy": spy}, n_outer=4, verbose=False)

    assert seen == [(False, False)]
    assert designer.training and action_net.training
    assert not designer.task.training


def test_evaluate_loss_checkpoint_rebuilds_the_run(tmp_path):
    task = _make_task()
    action_net = _make_action_net(task)
    cfg = {
        "T": T,
        "device": "cpu",
        "task": {
            "_target_": "testbed.tasks.LocationFinding",
            "n_sources": 1,
            "dim_theta": 2,
        },
        "designer": {"_target_": "testbed.designers.RandomDesigner"},
        "designer_run_dir": None,
        "action_net": {"_target_": "testbed.design_nets.DeepSetPolicyNet"},
        "final_eval": {"losses": LOSSES},
    }
    OmegaConf.save(OmegaConf.create(cfg), tmp_path / "config.yaml")
    torch.save(
        {"action_net": action_net.state_dict(), "use_raw_history": False},
        tmp_path / "final_step.pt",
    )

    out = evaluate_loss_checkpoint(str(tmp_path), n_outer=8, outer_batch_size=4)

    assert out["checkpoint"] == "final_step.pt"
    assert {"mse", "mse_se", "sorted_mse", "sorted_mse_se"} <= set(out)


def test_evaluate_loss_checkpoint_falls_back_to_mse(tmp_path):
    task = _make_task()
    cfg = {
        "T": T,
        "task": {
            "_target_": "testbed.tasks.LocationFinding",
            "n_sources": 1,
            "dim_theta": 2,
        },
        "designer": {"_target_": "testbed.designers.RandomDesigner"},
        "action_net": {"_target_": "testbed.design_nets.DeepSetPolicyNet"},
    }
    OmegaConf.save(OmegaConf.create(cfg), tmp_path / "config.yaml")
    torch.save(
        {"action_net": _make_action_net(task).state_dict()}, tmp_path / "final_step.pt"
    )

    out = evaluate_loss_checkpoint(str(tmp_path), n_outer=4, outer_batch_size=4)

    assert set(out) == {"mse", "mse_se", "n_outer", "checkpoint"}
