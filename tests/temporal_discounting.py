"""Full DAD training run on temporal discounting with the REINFORCE estimator.

Not collected by pytest (no test_ prefix); run manually:
    python tests/temporal_discounting.py

Trains a DAD policy with the REINFORCE gradient estimator (DAD paper Eq. 16),
then evaluates SPCE/SNMC bounds against a random-design baseline. Covers both
selectable policy nets: the outcome-gated encoder of the paper's HTD experiment
and the general-purpose concatenation DeepSets encoder.
"""

import time

import torch
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from testbed.design_nets import DeepSetPolicyNet, GatedDeepSetPolicyNet
from testbed.design_objectives import SNMC, SPCE, SPCEReinforce
from testbed.designers import PolicyDesigner, RandomDesigner
from testbed.tasks import HyperbolicTemporalDiscounting

# Reference sPCE for HTD at T=20: 5.021 +- 0.013, DAD paper (Foster et al. 2021).
# Note the paper trains for 100k steps; the short run here may fall slightly short.
EXPECTED_SPCE = 5.021
SPCE_TOL = 0.15

# Training (short GPU run; see recipes/experiments/temporal_discounting/dad.yaml for the full recipe)
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


def make_policy_designer(task, device, policy_cls):
    # task.design_space (an HTDDesignSpace) maps the net's raw output to valid
    # (D_1, R_0); no wrapper needed.
    design_net = policy_cls(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
        encoder_dim_h=[256, 256],
        dim_emb=16,
        decoder_dim_h=[128, 128],
    ).to(device)
    return PolicyDesigner(task, T, design_net)


def make_random_designer(task):
    # HTDDesignSpace.sample draws raw N(0, 1) and maps it to valid (D_1, R_0)
    # the same way as the policy net's wrapper (cf. Step-DAD random baseline).
    return RandomDesigner(task, T)


def train(task, designer):
    objective = SPCEReinforce(
        task,
        n_contrastive=N_INNER,
        stochastic_design_policy=False,  # deterministic policy net -> pathwise
    )
    optimizer = torch.optim.Adam(designer.design_net.parameters(), lr=LR)

    designer.design_net.train()
    for step in track(range(N_TRAIN_ITER), description="  Training DAD…"):
        theta0 = task.sample_prior(N_OUTER)
        rollout = designer.rollout(theta0)
        loss = -objective.estimate_gain(rollout)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(designer.design_net.parameters(), GRAD_CLIP)
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
            rollout = designer.rollout(theta0)
            spce_vals.append(spce.estimate_gain(rollout).item())
            snmc_vals.append(snmc.estimate_gain(rollout).item())

    return sum(spce_vals) / N_EVAL_RUNS, sum(snmc_vals) / N_EVAL_RUNS


# Both policy nets are usable from the config, so cover both here. The gated net
# reproduces the paper's HTD encoder (Appendix D.2); the concat net is the
# general-purpose DeepSets baseline.
POLICY_NETS = {
    "DAD gated (reinforce)": GatedDeepSetPolicyNet,
    "DAD concat (reinforce)": DeepSetPolicyNet,
}

# Only the paper-matching net is held to the paper's number.
REFERENCE_NET = "DAD gated (reinforce)"


def test_dad_reinforce_eig():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = HyperbolicTemporalDiscounting().to(device)

    results = {"Random": evaluate(task, make_random_designer(task))}
    train_min = 0.0
    for name, policy_cls in POLICY_NETS.items():
        # Same seed per net so the comparison isn't an initialization lottery.
        torch.manual_seed(0)
        designer = make_policy_designer(task, device, policy_cls)

        console.log(f"[bold]{name}[/bold]")
        start = time.perf_counter()
        train(task, designer)
        train_min += (time.perf_counter() - start) / 60

        designer.design_net.eval()
        results[name] = evaluate(task, designer)

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
                f"[green]Training ({len(POLICY_NETS)} nets): {train_min:.1f} min[/green]   |   "
                f"T={T}, n_train_iter={N_TRAIN_ITER:,}, n_contrastive={N_INNER_EVAL:,}, device={device}",
            ),
            title="EIG -- DAD trained with REINFORCE (HTD)",
            border_style="dim",
        )
    )

    random_spce = results["Random"][0]
    for name in POLICY_NETS:
        assert results[name][0] > random_spce, (
            f"{name} ({results[name][0]:.4f}) should beat random designs "
            f"({random_spce:.4f})"
        )

    reference_spce = results[REFERENCE_NET][0]
    if EXPECTED_SPCE is not None:
        assert abs(reference_spce - EXPECTED_SPCE) <= SPCE_TOL, (
            f"{REFERENCE_NET} sPCE {reference_spce:.4f} not within {SPCE_TOL} "
            f"of {EXPECTED_SPCE}"
        )


if __name__ == "__main__":
    test_dad_reinforce_eig()
