from argparse import ArgumentParser
from math import floor
from pathlib import Path
from time import time
from indago.model import (
    Diffrax,
    Dynamics_JAX,
    ParameterisedCellTransmissionModelNonSmooth_Jax,
    ParameterisedCellTransmissionModelSmooth_Jax,
    ParameterisedCellTransmissionModel_Numpy,
)
from scipy.integrate import solve_ivp
from semble.initial_state import (
    InitialStateGenerator,
    get_initial_state_generator,
)
from semble.sequence_generators import SequenceGenerator, get_sequence_generator
from semble.parameter_generators import (
    ParameterGenerator,
    get_parameter_generator,
)
from semble.dynamics import (
    ParameterisedCellTransmissionModel,
    GreenshieldsTraffic,
)
from flumen_jax import Flumen

import diffrax
import equinox
import jax
import jax.numpy as jnp
import jax.random as jrd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

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

SCIPY_ATOL = 1e-9
SCIPY_RTOL = 1e-9

NUMPY_RNG_SEED = 73577678


class GreenshieldsNumpy:
    def __init__(self, dynamics: GreenshieldsTraffic, delta: float):
        super().__init__()
        self.delta = delta
        self.v0 = dynamics.v0
        self.inv_step = dynamics.inv_step

    def __call__(self, t, y, u):
        n_control = int(floor(t / self.delta))
        u_val = u[n_control]

        q_out = self.v0 * y * (1.0 - y)
        q0_in = self.v0 * u_val * (1.0 - u_val)

        q_in = np.hstack((q0_in, q_out[:-1]))

        dy = self.inv_step * (q_in - q_out)
        return dy


class GreenshieldsJax:
    def __init__(self, dynamics: GreenshieldsTraffic, delta: float):
        super().__init__()
        self.v0 = dynamics.v0
        self.inv_step = dynamics.inv_step
        self.delta = delta

    def __call__(self, t, y, u_vec):
        index = jnp.floor(t / self.delta).astype(jnp.uint32)
        u_val = u_vec[index]
        q_out = self.v0 * y * (1.0 - y)
        q0_in = self.v0 * u_val * (1.0 - u_val)

        q_in = jnp.hstack((q0_in, q_out[:-1]))

        dy = self.inv_step * (q_in - q_out)

        return dy


def error_stats(y_true, y_other) -> tuple[float, float]:
    error = np.mean(
        np.linalg.norm(y_true - y_other, axis=-1) / np.linalg.norm(y_true),
        axis=1,
    )
    mean = np.mean(error, axis=0)
    std = np.std(error, axis=0)

    return mean.item(), std.item()


def solve_flumen_traj(flat_model, model_treedef, t, x0, u, delta, params):
    model: Flumen = jax.tree_util.tree_unflatten(model_treedef, flat_model)
    skips = jnp.floor(t / delta).astype(jnp.uint32)
    tau = (t - delta * skips) / delta

    return model.eval_trajectory(x0, u, tau, skips.squeeze(), params)


def solve_flumen_batch(flat_model, model_treedef, t, x0, u, delta, params):
    model: Flumen = jax.tree_util.tree_unflatten(model_treedef, flat_model)
    skips = jnp.floor(t / delta).astype(jnp.uint32)
    tau = (t - delta * skips) / delta
    skips = skips.squeeze()

    def eval(x0_, u_, tau_, skip, params_):
        tau_seq = jnp.ones((u_.shape[0], 1))
        tau_seq = tau_seq.at[skip, :].set(tau_)
        rnn_input = jnp.concatenate((u_, tau_seq), axis=-1)

        return model(x0_, rnn_input, tau_, skip + 1, params_)

    return jax.vmap(eval, in_axes=(None, None, 0, 0, None))(
        x0, u, tau, skips, params
    )


def sample_features(
    dynamics: ParameterisedCellTransmissionModel,
    init_state_gen: InitialStateGenerator,
    seq_gen: SequenceGenerator,
    par_gen: ParameterGenerator,
    n_samples: int,
    time_horizon: float,
    delta: float,
    rng,
):
    control_len = 1 + int(np.ceil(time_horizon / delta))

    x0 = np.empty((n_samples, dynamics.n), dtype=np.float64)
    u = np.empty((n_samples, control_len, dynamics.m), dtype=np.float64)
    params = np.empty((n_samples, par_gen.dim), dtype=np.float64)

    for k in range(n_samples):
        x0[k] = init_state_gen.sample(rng).astype(np.float64)
        u[k] = seq_gen.sample(
            time_range=(0, time_horizon), delta=delta, rng=rng
        ).astype(np.float32)
        params[k] = par_gen.sample(rng=rng).astype(np.float32)
    return x0, u, params


def compute_times_and_errors(
    flumen: Flumen,
    time_horizon: float,
    n_time_samples: int,
    n_warmup: int,
    dynamics: ParameterisedCellTransmissionModel,
    x0,
    u,
    params,
    delta: float,
    dt0=0.01,
    use_batched=False,
):
    def scipy_compute(x0, u, params, y, func):
        t = time()
        for k, (x_, u_, params_) in enumerate(
            zip(x0[n_warmup:], u[n_warmup:], params[n_warmup:])
        ):
            y[k] = func(x_, u_, params_)
        return (time() - t) / (x0.shape[0] - n_warmup)

    def warmup_and_time(x0, u, params, y, func):
        for x_, u_, params_ in zip(
            x0[:n_warmup], u[:n_warmup], params[:n_warmup]
        ):
            func(x_, u_, params_)
        t = time()
        for k, (x_, u_, params_) in enumerate(
            zip(x0[n_warmup:], u[n_warmup:], params[n_warmup:])
        ):
            y[k] = func(x_, u_, params_).block_until_ready()
        return (time() - t) / (x0.shape[0] - n_warmup)

    time_vector = np.linspace(0.0, time_horizon, n_time_samples)
    ts = diffrax.SaveAt(ts=time_vector)

    dynf_np = ParameterisedCellTransmissionModel_Numpy(dynamics, delta)

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

    y_scipy = np.empty((x0.shape[0] - n_warmup, dynamics.n, len(time_vector)))
    _ = scipy_compute(x0, u, params, y_scipy, scipy_func)
    y_scipy = np.transpose(y_scipy, axes=(0, 2, 1))

    # Smooth
    # dynf = ParameterisedCellTransmissionModelSmooth_Jax(dynamics, delta)
    # Non smooth
    dynf = ParameterisedCellTransmissionModelNonSmooth_Jax(dynamics, delta)
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()
    ode_term = diffrax.ODETerm(dynf)
    ts = diffrax.SaveAt(ts=time_vector)

    @jax.jit
    def diffrax_euler_func(x, u, params):
        return diffrax.diffeqsolve(
            ode_term,
            solver,
            t0=0.0,
            t1=time_horizon,
            dt0=dt0,
            y0=x,
            args=(u, params),
            saveat=ts,
            stepsize_controller=stepsize_controller,
            max_steps=None,
        ).ys

    y_diffrax_euler = np.empty_like(y_scipy)
    t_diffrax_euler = warmup_and_time(
        x0, u, params, y_diffrax_euler, diffrax_euler_func
    )

    mean_err_euler, std_err_euler = error_stats(y_scipy, y_diffrax_euler)

    euler_results = {
        "Method": "Euler",
        r"$T$": time_horizon,
        "Time per trajectory (s)": t_diffrax_euler,
        "Relative error (mean)": mean_err_euler,
        "Relative error (std)": std_err_euler,
    }

    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(atol=1e-3, rtol=1e-6)
    ode_term = diffrax.ODETerm(dynf)
    ts = diffrax.SaveAt(ts=time_vector)  # type: ignore

    @jax.jit
    def diffrax_tsit5_func(x, u, params):
        return diffrax.diffeqsolve(
            ode_term,
            solver,
            t0=0.0,
            t1=time_horizon,
            dt0=dt0,
            y0=x,
            args=(u, params),
            saveat=ts,
            stepsize_controller=stepsize_controller,
            # stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=None,
        ).ys

    y_diffrax_tsit5 = np.empty_like(y_scipy)
    t_diffrax_tsit5 = warmup_and_time(
        x0, u, params, y_diffrax_tsit5, diffrax_tsit5_func
    )

    mean_err_tsit5, std_err_tsit5 = error_stats(y_scipy, y_diffrax_tsit5)

    tsit5_results = {
        "Method": "Tsit5",
        r"$T$": time_horizon,
        "Time per trajectory (s)": t_diffrax_tsit5,
        "Relative error (mean)": mean_err_tsit5,
        "Relative error (std)": std_err_tsit5,
    }

    y_flumen = np.empty_like(y_scipy)
    time_vector = time_vector.reshape((-1, 1))
    flat_model, model_treedef = jax.tree_util.tree_flatten(flumen)

    solve_flumen = solve_flumen_batch if use_batched else solve_flumen_traj

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

    t_flumen = warmup_and_time(x0, u, params, y_flumen, flumen_func)
    mean_err_flumen, std_err_flumen = error_stats(y_scipy, y_flumen)

    flumen_results = {
        "Method": "Flumen",
        r"$T$": time_horizon,
        "Time per trajectory (s)": t_flumen,
        "Relative error (mean)": mean_err_flumen,
        "Relative error (std)": std_err_flumen,
    }

    return euler_results, tsit5_results, flumen_results


def main(args):
    # model_path = Path(args.path)
    import wandb

    api = wandb.Api()
    model_artifact = api.artifact(args.path)
    model_path = Path(model_artifact.download())

    with open(model_path / "metadata.yaml", "r") as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    like_model = equinox.filter_eval_shape(
        Flumen, **metadata["args"], key=jrd.key(0)
    )
    model: Flumen = equinox.tree_deserialise_leaves(
        model_path / "leaves.eqx", like_model
    )

    ds = metadata["data_settings"]
    dynamics = ParameterisedCellTransmissionModel(**ds["dynamics"]["args"])

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

    # avoid division by 0 so location U(0.1,0.9)
    ds["dynamics"]["args"]["parameter_generator"]["args"][0]["args"]["high"] = (
        0.9
    )
    ds["dynamics"]["args"]["parameter_generator"]["args"][0]["args"]["low"] = (
        0.1
    )

    par_gen = get_parameter_generator(
        ds["dynamics"]["args"]["parameter_generator"]["name"],
        ds["dynamics"]["args"]["parameter_generator"]["args"],
    )

    delta = metadata["data_settings"]["control_delta"]
    rng = np.random.default_rng(seed=NUMPY_RNG_SEED)

    all_results = []

    # compute 2 samples in advance to debias computation time
    n_warmup = 2

    for time_horizon in args.time_horizons:
        x0, u, params = sample_features(
            dynamics,
            init_state_gen,
            seq_gen,
            par_gen,
            args.n_traj_samples + n_warmup,
            time_horizon,
            delta,
            rng,
        )

        results = compute_times_and_errors(
            model,
            time_horizon,
            args.n_time_samples,
            n_warmup,
            dynamics,
            x0,
            u,
            params,
            delta,
            args.diffrax_dt0,
            args.use_batched,
        )

        all_results.extend(results)

    times_and_errors = pd.DataFrame(all_results)

    print(times_and_errors)

    _, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")

    # unique sorted T values
    # T_values = sorted(times_and_errors["$T$"].unique())

    # sample evenly spaced colors from viridis
    # colors = plt.cm.viridis(np.linspace(0, 1, len(T_values)))

    # map each T to a color
    # palette = dict(zip(T_values, colors))
    sns.scatterplot(
        times_and_errors,
        x="Time per trajectory (s)",
        y="Relative error (mean)",
        style="Method",
        hue=r"$T$",
        palette="colorblind",
        ax=ax,
    )
    plt.tight_layout()  # helps before saving
    plt.savefig("time_traj_eval_param_clb_smooth.pdf")
    plt.show()


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("path", type=str, help="Path to model folder.")
    ap.add_argument(
        "--n_traj_samples",
        type=int,
        help="Number of trajectories to sample.",
        default=100,
    )
    ap.add_argument(
        "--n_time_samples",
        type=int,
        help="Number of time eval points",
        default=10,
    )
    ap.add_argument(
        "--time_horizons",
        type=float,
        nargs="+",
        default=[20.0],
    )
    ap.add_argument("--use_batched", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--diffrax_dt0", type=float, default=0.01)

    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
