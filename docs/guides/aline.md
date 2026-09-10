# ALINE implementation notes

This implementation covers the parameter-target, fixed-candidate-pool variant of
[ALINE](https://arxiv.org/abs/2506.07259). It was compared against the authors’
checkout at commit `8aefb29f43721165a2a8a513cd1eec3b3904272f`.

Run with `python -m testbed.methods.aline`. The default recipe is
`testbed/recipes/method/aline.yaml`; candidate bounds belong to
`task.design_space`, which must be a finite, nondegenerate `BoxDesignSpace`.

## Reference behavior

- Context tokens attend to context; candidates attend to context and selected
  target tokens; target tokens attend only to context. Candidate and target
  self-attention is masked, as in `model/encoder.py`.
- The embedding MLPs, transformer, acquisition head, and marginal Gaussian
  mixture heads match the reference parameter-target architecture. Designs and
  outcomes have explicit center/scale settings, independent of candidate bounds.
- Inference trains all parameter marginals, including parameters outside the
  acquisition target subset. Training scores pre-acquisition states after the
  initial context, excluding the final post-acquisition state.
- Each acquisition except the last receives its detached next-state likelihood
  gain on the selected targets. Gains are clipped at zero, multiplied by
  `gamma**t` starting at `t=1`, then standardized across the batch with the sample
  standard deviation, before multiplication by action log probabilities.
- Warm-up samples the acquisition policy from a reduced pool and trains only
  inference. Joint training restarts the optimizer and cosine scheduler, reduces
  the shared/inference learning rate by five, and retains the acquisition head’s
  learning rate. Gradient clipping uses the infinity norm.
- Evaluation defaults to greedy acquisition and passes a TestBED `Rollout` to
  SPCE/SNMC. Simulation supplies the full history to the task API.

## TestBED conventions and limits

`T` counts all acquired observations, including `n_initial`; `pool_size` counts
all candidates, including the initial context. In the authors’ code, the initial
context is separate from its acquisition-loop length and query-pool size. Here,
at least two acquisitions after the initial context are required for a training
reward. The warm-up pool contains `T` total candidates. Fixed horizons are used;
the reference also supports randomly varying the training horizon.

The recipe retains TestBED’s Gaussian location-finding prior and configurable
outcome scaling. It does not reproduce the authors’ uniform-prior benchmark
configuration or their full training experiment. Predictive-data targets, mixed
data/parameter targets, and time tokens are outside this implementation’s scope.
For a singleton batch, reward standardization is omitted to avoid the reference’s
undefined sample standard deviation. Empty-context attention uses SDPA to keep
fully masked target rows finite in both training and evaluation.

The normalization buffers and changed attention semantics mean older ALINE
checkpoints should be retrained.

## Validation

`pytest` passes 554 tests with two unrelated skips on the local CPU environment.
This includes a two-step Hydra training/evaluation/checkpoint test, controlled
reward and history tests, and numerical forward/gradient parity checks loading
neural classes directly from the unmodified authors’ source. Those parity checks
use `../ALINE` by default, or `ALINE_REFERENCE_DIR`; they skip if the reference
checkout is absent.

HPC validation was attempted on `u6sb.aip2.isambard`, including elevated retries,
but SSH failed with `No route to host`. GPU execution and long-run learning
performance remain unverified.
