"""One-update GPU smoke test behind the certified pilot gate."""

import argparse
import os
import sys
from pathlib import Path

os.environ["ADAPTIVE_SCREENING_TRAINING"] = "1"
os.environ.setdefault("JAX_ENABLE_X64", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screening.config import load_config  # noqa: E402
from screening.gate import require_passing_gate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    require_passing_gate(config, args.gate)

    import jax
    import optax

    import screening.train as training
    from screening.neural import init_policy

    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU backend unavailable: {devices}")
    params = init_policy(jax.random.key(0))
    episodes = training.sample_training_episodes(config, jax.random.key(1), 32)
    examples = training.ucb_imitation_states(episodes, config, 2.8)
    optimizer = optax.adam(0.001)
    state = optimizer.init(params)
    params, state, bc_loss, _ = training._make_bc_step(config, 2.5, optimizer)(
        params, state, examples
    )
    reinforce = optax.adamw(0.0003)
    reinforce_state = reinforce.init(params)
    _, _, rl_loss, _, _, finite = training._make_reinforce_step(config, 2.5, reinforce)(
        params, reinforce_state, episodes, 0.01
    )
    if not bool(finite):
        raise FloatingPointError("GPU smoke update produced nonfinite state")
    print(
        f"GPU SMOKE PASS devices={devices} bc_loss={float(bc_loss):.6f} "
        f"reinforce_loss={float(rl_loss):.6f}"
    )


if __name__ == "__main__":
    main()
