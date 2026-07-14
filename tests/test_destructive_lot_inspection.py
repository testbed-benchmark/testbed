import torch

from testbed.envs import BEDEnv, RLBOEDDenseReward
from testbed.tasks import DestructiveLotInspection


def test_destructive_lot_inspection_wraps_with_bed_env():
    task = DestructiveLotInspection(num_zones=3)
    assert not task.discrete_theta
    env = BEDEnv(
        task,
        horizon=2,
        reward_fn=RLBOEDDenseReward(n_contrastive=2),
        batch_shape=(4,),
    )

    obs, info = env.reset(seed=123)
    assert info["t"] == 0
    assert obs.designs.shape == (4, 2)
    assert obs.outcomes.shape == (4, 2, 1)
    assert env.reward_fn.theta_all.shape == (4, 3, 3)

    obs, reward, terminated, truncated, info = env.step(torch.tensor([0, 1, 2, 1]))

    assert reward.shape == (4,)
    assert not terminated.any()
    assert not truncated.any()
    assert info["design"].dtype == torch.long
    assert torch.all(info["outcome"] == info["outcome"].round())
    assert torch.all((info["outcome"] >= 0.0) & (info["outcome"] <= task.batch_size))
    assert obs.mask.sum().item() == 4
