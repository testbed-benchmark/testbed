import time

import torch
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from testbed.core.distributions import TorchDistribution
from testbed.design_objectives import SNMC, SPCE
from testbed.designers import RandomDesigner
from testbed.tasks import LocationFinding

console = Console()


def test_random_eig():
    n_sources = 1
    dim_theta = 2
    T = 10
    n = 10
    n_contrastive = 50_000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    task = LocationFinding(n_sources, dim_theta, device=device)
    design_prior = TorchDistribution(
    torch.distributions.MultivariateNormal(
        torch.zeros(dim_theta, device=device),
        torch.eye(dim_theta, device=device),
    ))

    designer = RandomDesigner(task, T, design_prior)
    spce = SPCE(task, n_contrastive=n_contrastive)
    snmc = SNMC(task, n_contrastive=n_contrastive)

    table = Table(
        "Run", "Lower (SPCE)", "Upper (SNMC)", "Width",
        box=box.SIMPLE,
        show_footer=True,
        header_style="bold",
        footer_style="bold",
    )

    spce_vals, snmc_vals = [], []
    start = time.perf_counter()

    for i in track(range(n), description="  Estimating EIG…"):
        torch.manual_seed(i)
        with torch.no_grad():
            theta0 = task.prior.sample(1024)
            y_obs, xi_obs = designer.rollout(theta0)
            spce_val = spce.estimate_gain(y_obs, xi_obs, theta0).item()
            snmc_val = snmc.estimate_gain(y_obs, xi_obs, theta0).item()

        spce_vals.append(spce_val)
        snmc_vals.append(snmc_val)
        table.add_row(str(i + 1), f"{spce_val:.4f}", f"{snmc_val:.4f}", f"{snmc_val - spce_val:.4f}")

    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_spce = sum(spce_vals) / n
    avg_snmc = sum(snmc_vals) / n
    table.columns[1].footer = f"avg {avg_spce:.4f}"
    table.columns[2].footer = f"avg {avg_snmc:.4f}"
    table.columns[3].footer = f"avg {avg_snmc - avg_spce:.4f}"

    console.print(Panel(
        Group(
            table,
            f"Expected EIG ≈ 3.5-3.6   |   "
            f"[green]Runtime: {elapsed_ms / n:.1f} ms/run[/green]   |   "
            f"n_src={n_sources}, d_theta={dim_theta}, T={T}, n_contrastive={n_contrastive:,}, device={device}",
        ),
        title="Sanity check -- EIG (Random @ LF)",
        border_style="dim",
    ))

def test_logp():
    from stepdad.models.location_finding import (
        LocationFindingModel,
        LocationFindingPrior,
    )

    n_sources = 1
    dim_theta = 2
    n_timing_runs = 200
    n_samples = 10_000
    device = 'cpu'

    task = LocationFinding(n_sources, dim_theta, device=device)
    prior_stepdad = LocationFindingPrior(K=n_sources, p=dim_theta)
    task_stepdad = LocationFindingModel(prior_stepdad, design_net=None)

    # --- correctness check ---
    torch.manual_seed(0)
    test_y      = torch.randn((n_samples, 1), device=device)
    test_theta  = torch.randn((n_samples, 1, 2), device=device)
    test_design = torch.randn((n_samples, 2), device=device)

    logp         = task._outcome_logp(test_y, test_theta, test_design)
    logp_stepdad = task_stepdad.outcome_likelihood(
        test_theta.squeeze(1), test_design
    ).log_prob(test_y).squeeze()

    match    = torch.allclose(logp, logp_stepdad)
    max_err  = (logp - logp_stepdad).abs().max().item()
    mean_err = (logp - logp_stepdad).abs().mean().item()

    ctable = Table("Check", "Result", box=box.SIMPLE, show_header=True, header_style="bold")
    ctable.add_row("torch.allclose", "[green]✓ True[/green]" if match else "[red]✗ False[/red]")
    ctable.add_row("Max abs error",  f"{max_err:.2e}")
    ctable.add_row("Mean abs error", f"{mean_err:.2e}")

    # --- timing ---
    def time_fn(fn, n=n_timing_runs):
        times = []
        #for _ in track(range(n), description=f"  Timing {fn.__name__}…"):
        for _ in range(n):
            torch.manual_seed(0)
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        return sum(times) / n, min(times), max(times)

    def run_task():
        test_y      = torch.randn((n_samples, 1))
        test_theta  = torch.randn((n_samples, 1, 2))
        test_design = torch.randn((n_samples, 2))
        task._outcome_logp(test_y, test_theta, test_design)

    def run_stepdad():
        test_y      = torch.randn((n_samples, 1))
        test_theta  = torch.randn((n_samples, 1, 2))
        test_design = torch.randn((n_samples, 2))
        task_stepdad.outcome_likelihood(
            test_theta.squeeze(1), test_design
        ).log_prob(test_y)

    task_avg,    task_min,    task_max    = time_fn(run_task)
    stepdad_avg, stepdad_min, stepdad_max = time_fn(run_stepdad)
    speedup = stepdad_avg / task_avg
    faster  = "task" if task_avg < stepdad_avg else "stepdad"

    ttable = Table(
        "Implementation", "Mean (ms)", "Min (ms)", "Max (ms)",
        box=box.SIMPLE,
        header_style="bold",
        footer_style="bold",
    )
    ttable.add_row("TestBED logp",              f"{task_avg:.2f}",    f"{task_min:.2f}",    f"{task_max:.2f}")
    ttable.add_row("StepDAD logp", f"{stepdad_avg:.2f}", f"{stepdad_min:.2f}", f"{stepdad_max:.2f}")

    faster_str = "TestBED faster" if faster == "task" else "StepDAD faster"
    console.print(Panel(
        Group(
            ctable,
            ttable,
            f"[green]Speedup: {speedup:.2f} x [/green] [dim]({faster_str})[/dim]   |   "
            f"n={n_timing_runs} runs, n_samples={n_samples:,}, device={device}",
        ),
        title="Sanity check -- logp equivalence (LF)",
        border_style="dim",
    ))

if __name__ == "__main__":
    test_random_eig()
    test_logp()