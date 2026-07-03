"""Full DAD training run on temporal discounting with the REINFORCE estimator.

Not collected by pytest (no test_ prefix); run manually:
    python tests/temporal_discounting.py

Trains a DAD policy with the REINFORCE gradient estimator (DAD paper Eq. 16),
then evaluates SPCE/SNMC bounds against a random-design baseline.
"""

import time

import torch
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from testbed.core.distributions import TorchDistribution
from testbed.design_nets import DeepSetPolicyNet
from testbed.design_objectives import SNMC, SPCE, Reinforce
from testbed.designers import PolicyDesigner, RandomDesigner
from testbed.tasks import HyperbolicTemporalDiscounting

# Reference sPCE for HTD at T=20 (from the literature). None skips the check.
EXPECTED_SPCE = None  # TODO: fill in reference value + source
SPCE_TOL = 0.1

# Training (short GPU run; see recipes/temporal_discounting/dad.yaml for the full recipe)
T = 20
N_TRAIN_ITER = 5_000
N_OUTER = 512
N_INNER = 511
LR = 5e-5
GRAD_CLIP = 1.0

# Evaluation
N_EVAL_RUNS = 5
N_OUTER_EVAL = 512
N_INNER_EVAL = 50_000

console = Console()


def make_policy_designer(task, device):
    policy_net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
        encoder_dim_h=256,
        encoder_layers=2,
        dim_emb=16,
        decoder_layers=2,
        decoder_dim_h=128,
    ).to(device)
    return PolicyDesigner(task, T, policy_net)


def make_random_designer(task, device):
    # Standard normal in the unconstrained design space (cf. Step-DAD random baseline)
    design_prior = TorchDistribution(
        torch.distributions.Normal(
            torch.zeros(2, device=device),
            torch.ones(2, device=device),
        )
    )
    return RandomDesigner(task, T, design_prior)


def train(task, designer):
    objective = Reinforce(
        task,
        reward_fn=SPCE(task, n_contrastive=N_INNER).per_sample_gain,
        baseline=None,
        differentiable_reward=True,
    )
    optimizer = torch.optim.Adam(designer.policy_net.parameters(), lr=LR)

    designer.policy_net.train()
    for step in track(range(N_TRAIN_ITER), description="  Training DAD…"):
        theta0 = task.sample_prior(N_OUTER)
        y_obs, design_obs = designer.rollout(theta0)
        loss = -objective.estimate_gain(y_obs, design_obs, theta0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(designer.policy_net.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % (N_TRAIN_ITER // 10) == 0:
            console.log(f"step {step}: gain={-loss.item():.4f}")


def evaluate(task, designer):
    spce = SPCE(task, n_contrastive=N_INNER_EVAL)
    snmc = SNMC(task, n_contrastive=N_INNER_EVAL)

    spce_vals, snmc_vals = [], []
    for i in range(N_EVAL_RUNS):
        torch.manual_seed(i)
        with torch.no_grad():
            theta0 = task.prior.sample(N_OUTER_EVAL)
            y_obs, xi_obs = designer.rollout(theta0)
            spce_vals.append(spce.estimate_gain(y_obs, xi_obs, theta0).item())
            snmc_vals.append(snmc.estimate_gain(y_obs, xi_obs, theta0).item())

    return sum(spce_vals) / N_EVAL_RUNS, sum(snmc_vals) / N_EVAL_RUNS


def test_dad_reinforce_eig():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    task = HyperbolicTemporalDiscounting(unconstrained_design=True).to(device)

    random_designer = make_random_designer(task, device)
    policy_designer = make_policy_designer(task, device)

    start = time.perf_counter()
    train(task, policy_designer)
    train_min = (time.perf_counter() - start) / 60

    policy_designer.policy_net.eval()
    results = {
        "Random": evaluate(task, random_designer),
        "DAD (reinforce)": evaluate(task, policy_designer),
    }

    table = Table(
        "Designer",
        "Lower (SPCE)",
        "Upper (SNMC)",
        "Width",
        box=box.SIMPLE,
        header_style="bold",
    )
    for name, (spce_val, snmc_val) in results.items():
        table.add_row(
            name, f"{spce_val:.4f}", f"{snmc_val:.4f}", f"{snmc_val - spce_val:.4f}"
        )

    expected = (
        f"Expected sPCE ≈ {EXPECTED_SPCE} ± {SPCE_TOL}"
        if EXPECTED_SPCE is not None
        else "Expected sPCE: not set"
    )
    console.print(
        Panel(
            Group(
                table,
                f"{expected}   |   "
                f"[green]Training: {train_min:.1f} min[/green]   |   "
                f"T={T}, n_train_iter={N_TRAIN_ITER:,}, n_contrastive={N_INNER_EVAL:,}, device={device}",
            ),
            title="EIG -- DAD trained with REINFORCE (HTD)",
            border_style="dim",
        )
    )

    dad_spce = results["DAD (reinforce)"][0]
    random_spce = results["Random"][0]
    assert dad_spce > random_spce, (
        f"Trained policy ({dad_spce:.4f}) should beat random designs ({random_spce:.4f})"
    )
    if EXPECTED_SPCE is not None:
        assert abs(dad_spce - EXPECTED_SPCE) <= SPCE_TOL, (
            f"Trained sPCE {dad_spce:.4f} not within {SPCE_TOL} of {EXPECTED_SPCE}"
        )


if __name__ == "__main__":
    test_dad_reinforce_eig()
