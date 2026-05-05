from pathlib import Path

import os
import equinox
import jax
import jax.numpy as jnp
import jax.random as jrd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from flumen_jax import Flumen
from scipy.integrate import solve_ivp
from semble.dynamics import (
    VanDerPolParameterised,
)
from semble.initial_state import (
    InitialStateGenerator,
    get_initial_state_generator,
)

from semble.sequence_generators import SequenceGenerator, get_sequence_generator

from indago.model import (
    ParameterisedVanDerPol_Numpy,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
    }
)

NUMPY_RNG_SEED = 6003550917
SCIPY_ATOL = 1e-9
SCIPY_RTOL = 1e-9


def solve_flumen_traj(flat_model, model_treedef, t, x0, u, delta, params):
    model: Flumen = jax.tree_util.tree_unflatten(model_treedef, flat_model)
    skips = jnp.floor(t / delta).astype(jnp.uint32)
    tau = (t - delta * skips) / delta

    return model.eval_trajectory(x0, u, tau, skips.squeeze(), params)


def sample_features(
    dynamics: VanDerPolParameterised,
    init_state_gen: InitialStateGenerator,
    seq_gen: SequenceGenerator,
    n_samples: int,
    time_horizon: float,
    delta: float,
    rng,
):
    control_len = 1 + int(np.ceil(time_horizon / delta))

    x0 = np.empty((n_samples, dynamics.n), dtype=np.float64)
    u = np.empty((n_samples, control_len, dynamics.m), dtype=np.float64)

    params = np.array([0.0, 1.0, 1.0, 2.0, 2.5, 3.5], dtype=np.float32).reshape(
        n_samples, 1
    )

    for k in range(n_samples):
        x0[k] = init_state_gen.sample(rng).astype(np.float64)
        u[k] = seq_gen.sample(
            time_range=(0, time_horizon), delta=delta, rng=rng
        ).astype(np.float32)

    return x0, u, params


def compute_trajectories(
    flumen: Flumen,
    time_horizon: float,
    n_time_samples: int,
    dynamics: VanDerPolParameterised,
    x0,
    u,
    params,
    delta: float,
):

    def compute_trajectories(x0, u, params, y, func):
        for k, (x_, u_, params_) in enumerate(zip(x0, u, params)):
            y[k] = func(x_, u_, params_)

    time_vector = np.linspace(0.0, time_horizon, n_time_samples)
    dynf_np = ParameterisedVanDerPol_Numpy(VanDerPolParameterised, delta)

    def scipy_func(x, u, params):
        return solve_ivp(
            dynf_np,
            (0.0, time_horizon),
            x,
            t_eval=time_vector,
            args=(u, params),
            method="RK45",
            atol=SCIPY_ATOL,
            rtol=SCIPY_RTOL,
        ).y

    y_scipy = np.empty((x0.shape[0], dynamics.n, len(time_vector)))
    compute_trajectories(x0, u, params, y_scipy, scipy_func)
    y_scipy = np.transpose(y_scipy, axes=(0, 2, 1))

    y_flumen = np.empty_like(y_scipy)
    time_vector = time_vector.reshape((-1, 1))
    flat_model, model_treedef = jax.tree_util.tree_flatten(flumen)

    solve_flumen = solve_flumen_traj

    @jax.jit
    def flumen_func(x, u, params):
        return solve_flumen(
            flat_model,
            model_treedef,
            time_vector,
            x,
            u,
            delta,
            params,
        )

    compute_trajectories(x0, u, params, y_flumen, flumen_func)
    return time_vector, y_scipy, y_flumen


def main():
    path = "models/vdp/"
    model_path = Path(path)

    with open(model_path / "metadata.yaml", "r") as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    like_model = equinox.filter_eval_shape(
        Flumen, **metadata["args"], key=jrd.key(0)
    )
    model: Flumen = equinox.tree_deserialise_leaves(
        model_path / "leaves.eqx", like_model
    )

    ds = metadata["data_settings"]
    dynamics = VanDerPolParameterised(**ds["dynamics"]["args"])

    seq_gen = get_sequence_generator(
        ds["sequence_generator"]["name"],
        ds["sequence_generator"]["args"],
    )

    if "initial_state_generator" in ds:
        init_state_gen = get_initial_state_generator(
            ds["initial_state_generator"]["name"],
            ds["initial_state_generator"]["args"],
        )
    else:
        init_state_gen = dynamics.default_initial_state()

    delta = metadata["data_settings"]["control_delta"]
    rng = np.random.default_rng(seed=NUMPY_RNG_SEED)
    time_horizon = metadata["data_args"]["time_horizon"]

    n_trajectories = 6
    x0, u, params = sample_features(
        dynamics,
        init_state_gen,
        seq_gen,
        n_trajectories,
        time_horizon,
        delta,
        rng,
    )

    n_time_samples = 200
    t, y_true, y_pred = compute_trajectories(
        model, time_horizon, n_time_samples, dynamics, x0, u, params, delta
    )

    save_dir = "vdp_trajectories"
    os.makedirs(save_dir, exist_ok=True)  # creates folder if it doesn't exist
    for trajectory in range(0, n_trajectories):
        fig, ax = plt.subplots(3, 1, sharex=True)
        for k, ax_ in enumerate(ax[: y_true.shape[-1]]):
            ax_.plot(
                t, y_pred[trajectory, :, k], c="orange", label="Prediction"
            )
            ax_.plot(t, y_true[trajectory, :, k], "b--", label="True state")

            ### Paper visualization purposes
            if trajectory in [0, 2, 4]:
                if trajectory == 0:
                    ax[0].legend()
                ax_.set_ylabel(f"$x_{k + 1}$")

        t_u = np.linspace(0.0, time_horizon, len(u[trajectory]))
        ax[-1].step(t_u, u[trajectory].squeeze(), where="post")
        ax[-1].set_xlabel("$t$")
        ax[0].set_title(f"$\\theta = {params[trajectory].item()}$")

        ### Paper visualization purposes
        if trajectory in [0, 2, 4]:
            ax[-1].set_ylabel("$u$")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.savefig(f"vdp_trajectories/trajectory_{trajectory}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
