from argparse import ArgumentParser
from pathlib import Path
from random import uniform
from time import time

import os
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
from flumen_jax import Flumen
from scipy.integrate import solve_ivp
from semble.dynamics import (
    ParameterisedCellTransmissionModel,
)
from semble.initial_state import (
    InitialStateGenerator,
    get_initial_state_generator,
)
from semble.parameter_generators import (
    ParameterGenerator,
    get_parameter_generator,
)
from semble.sequence_generators import SequenceGenerator, get_sequence_generator

from indago.model import (
    ParameterisedCellTransmissionModel_Jax,
    ParameterisedCellTransmissionModel_Numpy,
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
SCIPY_ATOL = 1e-9
SCIPY_RTOL = 1e-9

NUMPY_RNG_SEED = 6003550914


def error(y_true, y_other):
    error = np.sqrt(
        np.mean(np.sum((y_true - y_other) ** 2, axis=-1), axis=-1)
        / np.mean(np.sum(y_true**2, axis=-1), axis=-1)
    )
    return error


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
    dts: list,
    use_batched=False,
):
    dts.sort(reverse=True)

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

    y_flumen = np.empty_like(y_scipy)
    time_vector_flumen = time_vector.reshape((-1, 1))
    flat_model, model_treedef = jax.tree_util.tree_flatten(flumen)

    solve_flumen = solve_flumen_batch if use_batched else solve_flumen_traj

    @jax.jit
    def flumen_func(x, u, params):
        return solve_flumen(
            flat_model,
            model_treedef,
            time_vector_flumen,
            x,
            u,
            delta,
            params,
        )

    t_flumen = warmup_and_time(x0, u, params, y_flumen, flumen_func)
    error_flumen = error(y_scipy, y_flumen)

    flumen_results = {
        "Method": "Flumen",
        r"$T$": time_horizon,
        "Time per trajectory (s)": t_flumen,
        "Relative error": error_flumen,
    }

    euler_results = []
    dynf_jax = ParameterisedCellTransmissionModel_Jax(dynamics, delta)
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()
    ode_term = diffrax.ODETerm(dynf_jax)
    ts = diffrax.SaveAt(ts=time_vector)  # type: ignore

    for dt in dts:

        @jax.jit
        def euler_func(x, u, params):
            return diffrax.diffeqsolve(
                ode_term,
                solver,
                t0=0.0,
                t1=time_horizon,
                dt0=dt,
                y0=x,
                args=(u, params),
                saveat=ts,
                stepsize_controller=stepsize_controller,
                max_steps=100 + int(time_horizon / dt),
            ).ys

        y_euler = np.empty_like(y_scipy)
        t_euler = warmup_and_time(x0, u, params, y_euler, euler_func)
        error_euler = error(y_scipy, y_euler)
        euler_results.append(
            {
                "Method": f"Euler (dt={dt})",
                r"$T$": time_horizon,
                "Time per trajectory (s)": t_euler,
                "Relative error": error_euler,
            }
        )

    @jax.jit
    def diffrax_tsit5_func(x, u, params):
        return diffrax.diffeqsolve(
            ode_term,
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=time_horizon,
            dt0=dts[-1],
            y0=x,
            args=(u, params),
            saveat=ts,
            stepsize_controller=diffrax.PIDController(atol=1e-6, rtol=1e-9),
        ).ys

    y_diffrax_tsit5 = np.empty_like(y_scipy)
    t_diffrax_tsit5 = warmup_and_time(
        x0, u, params, y_diffrax_tsit5, diffrax_tsit5_func
    )

    tsit5_error = error(y_scipy, y_diffrax_tsit5)

    tsit5_results = {
        "Method": "Tsit5",
        r"$T$": time_horizon,
        "Time per trajectory (s)": t_diffrax_tsit5,
        "Relative error": tsit5_error,
    }
    return flumen_results, *euler_results, tsit5_results


def main(args):
    model_path = Path("models/ctm/")

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

    # Make sure velocity is upper bounded for Euler's method.
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
            args.dts,
            args.use_batched,
        )

        all_results.extend(results)

    times_and_errors = pd.DataFrame(all_results)
    times = times_and_errors["Time per trajectory (s)"]
    print(times_and_errors)

    times_and_errors = times_and_errors.explode("Relative error")

    # jitter times to make it look nicer
    times_and_errors["Time per trajectory (s)"] = times_and_errors[
        "Time per trajectory (s)"
    ].apply(lambda x: x * (1 + uniform(-0.1, 0.1)))

    _, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")

    sns.scatterplot(
        times_and_errors,
        x="Time per trajectory (s)",
        y="Relative error",
        hue="Method",
        palette="colorblind",
        ax=ax,
    )
    for t in times:
        ax.axvline(x=t, alpha=0.2)
    plt.tight_layout()  # helps before saving
    save_dir = "results/timings/ctm"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/traj_timings.pdf")
    plt.show()


def parse_args():
    ap = ArgumentParser()
    ap.add_argument(
        "--n_traj_samples",
        type=int,
        help="Number of trajectories to sample.",
        default=50,
    )
    ap.add_argument(
        "--n_time_samples",
        type=int,
        help="Number of time eval points",
        default=100,
    )
    ap.add_argument(
        "--time_horizons",
        type=float,
        nargs="+",
        default=[25.0],
    )
    ap.add_argument("--use_batched", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--dts", type=float, nargs="+", default=[0.005])

    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
