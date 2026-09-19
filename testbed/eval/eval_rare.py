r"""sPCE/sNMC evaluation restricted to the rare tail of the prior.

The headline bounds in `eval_policy` are expectations under the full prior `p`,
so a policy that is excellent on typical theta and useless on atypical theta
still scores well. Here the outer expectation is taken over

    q(theta) \propto p(theta) * 1[log p(theta) < tau]

instead, which asks what a policy learns when theta is a rare draw.

The quantity being bounded is the expected information gain under the original
model `p`, with the outer expectation conditioned on theta being in the rare
set. The two bounds on it need *different* contrastive measures:

    sPCE_{L,q}  <=  I_q(theta; Y)  <=  the conditioned EIG  <=  NMC_{L,p}

- lower: outer theta_0 ~ q, contrastives ~ q. Its marginal tends to p_q(y), so
  it targets the rare-region EIG.
- upper: outer theta_0 ~ q, contrastives ~ p. Its marginal tends to p_p(y), so
  it targets I_q(theta; Y) + KL(p_q(y) || p_p(y)) -- the rare-region EIG plus a
  term for how atypical the rare-region predictive is.

That is why `rare_spce_snmc` runs two contrastive passes rather than sharing
one set of draws the way `chunked_spce_snmc` does, and why this costs about
twice the `history_logp` work of an ordinary evaluation. Rejection sampling
adds to that: it accepts a `percentile / 100` fraction, so the theta draws
themselves cost `100 / percentile` times more.

`tau` is fixed before any evaluation draw is taken, either from a seeded pilot
sample at a requested percentile or pinned directly in the config. Fixing it up
front is what keeps the draws i.i.d.; taking the bottom percentile of each
evaluation batch by ordering would correlate them. The seed is what makes the
comparison fair, since every method then sees the same rare set.

Usage:
    python -m testbed.eval.eval_rare <run_dir>

<run_dir> is a training run's output directory. Everything comes from its saved
`rare_eval` block: the rare set, the sizes and the checkpoint. `final_eval` is
not consulted, so a rare evaluation can be sized independently of the headline
one it is usually run alongside.
"""

import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from testbed.core.distributions import Distribution
from testbed.design_objectives import SNMC, SPCE
from testbed.design_objectives.contrastive import contrastive_logsumexps
from testbed.eval.base import eval_spce_snmc
from testbed.eval.eval_policy import load_policy_designer

# Proposals for the first rejection round, before an acceptance rate is known.
INITIAL_OVERSAMPLE = 4
# Headroom on later rounds, so a rate estimated from few draws still converges.
OVERSAMPLE_MARGIN = 1.2
# Ceiling on one round's proposal tensor. Without it, a deep tail turns a large
# request into a single enormous allocation. Capping the round keeps memory flat
# and just loops more.
MAX_PROPOSE_PER_ROUND = 1_000_000
# Total proposals allowed per sample() call, relative to what was asked for. A
# threshold far enough into the tail to exceed this is a configuration error,
# not something to grind away at.
MAX_PROPOSE_FACTOR = 100_000
# torch.quantile rejects inputs beyond this many elements.
QUANTILE_LIMIT = 16_000_000
# The recipe group holding the rare-tail settings, read at evaluation time.
RARE_EVAL_DEFAULTS = (
    Path(__file__).resolve().parents[1] / "recipes" / "rare_eval" / "default.yaml"
)


class TailTruncated(Distribution):
    """A distribution restricted to where its base has log-density below a cutoff.

    Sampling is by rejection against a threshold fixed at construction, so the
    draws are i.i.d. from the truncated distribution. The acceptance rate is not
    supplied: rounds repeat until enough have been accepted, sizing each one
    from the rate measured so far, which is what lets a hand-pinned threshold be
    used without knowing what percentile it is.
    """

    def __init__(self, base: Distribution, threshold):
        """Wraps `base`, keeping only theta with `base.log_prob(theta) < threshold`."""
        super().__init__()
        self.base = base
        self.register_buffer("threshold", torch.as_tensor(threshold))
        self.n_accepted = 0
        self.n_proposed = 0

    @property
    def realised_rate(self) -> float:
        """Fraction of proposals accepted so far, or nan before any sampling."""
        if self.n_proposed == 0:
            return float("nan")
        return self.n_accepted / self.n_proposed

    def _propose(self, n_wanted: int) -> torch.Tensor:
        """Draws one capped round from the base and returns the accepted theta."""
        rate = self.realised_rate
        if math.isnan(rate) or rate <= 0.0:
            n_propose = n_wanted * INITIAL_OVERSAMPLE
        else:
            n_propose = math.ceil(n_wanted / rate * OVERSAMPLE_MARGIN)
        n_propose = min(n_propose, MAX_PROPOSE_PER_ROUND)
        theta = self.base.sample(n_samples=n_propose)
        accepted = theta[self.base.log_prob(theta) < self.threshold]
        self.n_proposed += n_propose
        self.n_accepted += accepted.shape[0]
        return accepted

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Draws `n_samples` from the truncated distribution by rejection."""
        budget = n_samples * MAX_PROPOSE_FACTOR
        collected, n_have, n_spent = [], 0, 0
        while n_spent < budget:
            before = self.n_proposed
            accepted = self._propose(n_samples - n_have)
            n_spent += self.n_proposed - before
            collected.append(accepted)
            n_have += accepted.shape[0]
            if n_have >= n_samples:
                return torch.cat(collected)[:n_samples]
        raise RuntimeError(
            f"only drew {n_have} of {n_samples} samples below threshold "
            f"{self.threshold.item()} after {n_spent} proposals (acceptance "
            f"rate {self.realised_rate}); the threshold is likely far into the "
            f"tail of the base distribution"
        )

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Base log-density renormalised by the acceptance rate, -inf above the cutoff.

        The normalising constant is only as good as the acceptance rate measured
        so far. Nothing on the sPCE/sNMC path calls this -- `contrastive_logsumexps`
        only ever samples -- so it is here for the Distribution interface.
        """
        base_logp = self.base.log_prob(theta)
        return torch.where(
            base_logp < self.threshold,
            base_logp - math.log(self.realised_rate),
            torch.full_like(base_logp, -float("inf")),
        )


def _pilot_threshold(prior, percentile, n_pilot, pilot_seed):
    """Estimates the log-density cutoff at `percentile` from a seeded pilot sample.

    The global RNG is saved and restored around the draw: task priors sample
    through it, so seeding it is the only way to fix the pilot, and leaving it
    seeded would also fix the evaluation's own draws.
    """
    if n_pilot > QUANTILE_LIMIT:
        raise SystemExit(
            f"rare_eval.n_pilot={n_pilot} exceeds torch.quantile's "
            f"{QUANTILE_LIMIT}-element limit"
        )
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(pilot_seed)
        with torch.no_grad():
            pilot = prior.sample(n_samples=n_pilot)
            log_probs = prior.log_prob(pilot)
    finally:
        torch.set_rng_state(rng_state)
    return torch.quantile(log_probs, percentile / 100.0)


def rare_tail_prior(
    prior, percentile=None, threshold=None, n_pilot=1_000_000, pilot_seed=0
):
    """Builds the rare-tail restriction of `prior`.

    Exactly one of `percentile` and `threshold` must be given. A threshold is
    used as-is and draws no pilot sample; a percentile is turned into one by a
    seeded pilot draw.
    """
    if (percentile is None) == (threshold is None):
        raise SystemExit(
            "set exactly one of rare_eval.percentile / rare_eval.threshold "
            f"(got percentile={percentile}, threshold={threshold})"
        )
    if threshold is None:
        threshold = _pilot_threshold(prior, percentile, n_pilot, pilot_seed)
    return TailTruncated(prior, threshold)


def rare_spce_snmc(
    task, rollout, n_contrastive, chunk_size, fused=True, *, rare_prior, base_prior
):
    """Per-sample sPCE_{L,q} (lower) and NMC_{L,p} (upper) gains for a rare rollout.

    Unlike `chunked_spce_snmc`, the two bounds cannot share one set of
    contrastive draws: the lower bound contrasts against the rare prior and the
    upper against the original one. `contrastive_logsumexps` draws from
    `task.current_prior`, so the two passes are separated by `update_prior`,
    which is left back at `base_prior`. Only `current_prior` is touched;
    `task.prior` is not.

    The second pass recomputes `logp_joint` and discards it. That costs `b`
    likelihood evaluations against `b * n_contrastive` for the draws it is
    bundled with, which is not worth a second code path to avoid.
    """
    task.update_prior(rare_prior)
    logp_joint, rare_logsumexps = contrastive_logsumexps(
        task, rollout, n_contrastive, chunk_size, fused=fused
    )
    task.update_prior(base_prior)
    _, base_logsumexps = contrastive_logsumexps(
        task, rollout, n_contrastive, chunk_size, fused=fused
    )
    lower = SPCE.gain_from_logsumexps(logp_joint, rare_logsumexps, n_contrastive)
    upper = SNMC.gain_from_logsumexps(logp_joint, base_logsumexps, n_contrastive)
    return lower, upper


def rare_eval_kwargs(task, percentile=None, threshold=None, **kwargs):
    """Builds the `eval_spce_snmc` arguments that restrict an evaluation to the tail.

    Returns `(eval_kwargs, tail)`: the keyword arguments to pass through to
    `eval_spce_snmc`, and the `TailTruncated` prior itself, which
    `rare_metrics` reads the realised percentile off once evaluation is done.
    """
    base_prior = task.prior
    tail = rare_tail_prior(base_prior, percentile, threshold, **kwargs)

    def bounds_fn(task, rollout, n_contrastive, chunk_size, fused=True):
        return rare_spce_snmc(
            task,
            rollout,
            n_contrastive,
            chunk_size,
            fused=fused,
            rare_prior=tail,
            base_prior=base_prior,
        )

    eval_kwargs = {"theta_sampler": tail.sample, "bounds_fn": bounds_fn}
    return eval_kwargs, tail


def rare_metrics(tail: TailTruncated, percentile=None) -> dict:
    """The threshold an evaluation ran at, and the percentile it corresponds to.

    Exactly one of the two was configured and the other derived, so they always
    describe the same rare set. Call this after evaluating, so a percentile
    derived from a pinned threshold covers every draw actually taken.
    """
    if percentile is None:
        percentile = 100.0 * tail.realised_rate
    return {
        "rare_log_prob_threshold": tail.threshold.item(),
        "rare_percentile": float(percentile),
    }


# The fields of a rare_eval block that describe the rare set itself, as opposed
# to how much of it to evaluate. Only these reach `rare_eval_kwargs`.
TAIL_KEYS = ("percentile", "threshold", "n_pilot", "pilot_seed")


def rare_settings(percentile=None, threshold=None, **overrides):
    """The rare-tail settings, from the recipe group with `overrides` applied.

    Read from `recipes/rare_eval/default.yaml`: a rare evaluation is done to a
    finished run and every re-eval of every method reads the same
    file.

    `percentile` and `threshold` are a pair, since exactly one may be set.
    Giving either replaces both, so asking for a threshold on the command line
    cannot collide with the percentile the file happens to carry.
    """
    settings = OmegaConf.load(RARE_EVAL_DEFAULTS)
    if percentile is not None or threshold is not None:
        settings.percentile = percentile
        settings.threshold = threshold
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return OmegaConf.to_container(settings, resolve=True)


def tail_settings(block: dict) -> dict:
    """Narrows a rare_eval block to the fields describing the rare set."""
    return {key: block[key] for key in TAIL_KEYS if key in block}


def evaluate_rare_policy_checkpoint(
    run_dir: str, rare: dict, device: str = "cpu"
) -> dict:
    """Scores a policy checkpoint on the rare tail, driven by a `rare_eval` block.

    A rare evaluation is its own evaluation rather than a variant of the
    headline one, so the sizes, the checkpoint and the designer all come from
    `rare_eval` and `final_eval` is not consulted. It is usually run smaller:
    the two contrastive passes roughly double the likelihood work, and
    rejection sampling multiplies the theta draws by `100 / percentile`.
    """
    task, designer, key = load_policy_designer(
        run_dir,
        checkpoint=rare["checkpoint"],
        use_raw_weights=rare["use_raw_weights"],
        device=device,
        designer_config=OmegaConf.create(rare["designer"]),
    )
    eval_kwargs, tail = rare_eval_kwargs(task, **tail_settings(rare))

    out = eval_spce_snmc(
        task,
        designer,
        n_outer=rare["n_outer"],
        n_contrastive=rare["n_contrastive"],
        outer_batch_size=rare["outer_batch_size"],
        contrastive_chunk_size=rare["contrastive_chunk_size"],
        fused=rare["fused"],
        **eval_kwargs,
    )
    out["weights_used"] = key
    out.update(rare_metrics(tail, rare.get("percentile")))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m testbed.eval.eval_rare <run_dir>")

    run_dir = Path(sys.argv[1])
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    print(
        evaluate_rare_policy_checkpoint(
            str(run_dir), rare_settings(), device=cfg.device
        )
    )
