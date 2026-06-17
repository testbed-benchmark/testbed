import pytest
import torch

from testbed.envs import BEDEnv, RLBOEDDenseReward
from testbed.tasks import DestructiveLotInspection, DestructiveLotInspectionPrior


def test_destructive_lot_inspection_prior_samples_and_scores():
    prior = DestructiveLotInspectionPrior(
        num_zones=3,
        prior_a=torch.tensor([1.0, 2.0, 3.0]),
        prior_b=4.0,
    )

    samples = prior.sample(5)
    logp = prior.log_prob(torch.full((2, 3), 0.5))

    assert samples.shape == (5, 3)
    assert torch.all((samples >= 0.0) & (samples <= 1.0))
    assert logp.shape == (2,)
    with pytest.raises(ValueError, match="shape"):
        DestructiveLotInspectionPrior(num_zones=3, prior_a=torch.ones(2))


def test_destructive_lot_inspection_logp_selects_design_zone():
    task = DestructiveLotInspection(num_zones=3, prior_a=1.0, prior_b=1.0)
    theta = torch.tensor([[0.2, 0.8, 0.5], [0.2, 0.8, 0.5]])
    design = torch.tensor([0, 1])
    outcome = torch.tensor([[1.0], [0.0]])

    logp = task.outcome_logp(outcome, theta, design)

    assert torch.allclose(logp, torch.log(torch.tensor([0.2, 0.2])))


def test_destructive_lot_inspection_samples_binary_outcomes():
    task = DestructiveLotInspection(num_zones=2)
    theta = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    design = torch.tensor([0, 1])

    outcome = task.sample_outcome(theta, design)

    assert outcome.shape == (2, 1)
    assert torch.equal(outcome, torch.tensor([[0.0], [1.0]]))


def test_destructive_lot_inspection_wraps_with_bed_env():
    task = DestructiveLotInspection(num_zones=3)
    env = BEDEnv(
        task,
        horizon=2,
        reward_fn=RLBOEDDenseReward(n_contrastive=2),
        batch_shape=(4,),
    )

    obs, info = env.reset(seed=123)
    assert info["t"] == 0
    assert obs["designs"].shape == (4, 2)
    assert obs["outcomes"].shape == (4, 2, 1)
    assert env.theta_all.shape == (4, 3, 3)

    obs, reward, terminated, truncated, info = env.step(torch.tensor([0, 1, 2, 1]))

    assert reward.shape == (4,)
    assert not terminated.any()
    assert not truncated.any()
    assert info["design"].dtype == torch.long
    assert torch.all((info["outcome"] == 0.0) | (info["outcome"] == 1.0))
    assert obs["mask"].sum().item() == 4
