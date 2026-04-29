from argparse import ArgumentParser
from pathlib import Path
from time import time
from typing import cast
from random import uniform
from indago.model import (
    ParameterisedCellTransmissionModel_Jax,
)
from jaxtyping import Array
from semble.dynamics import (
    ParameterisedCellTransmissionModel,
)
from semble.parameter_generators import (
    ParameterGenerator,
    get_parameter_generator,
)
from semble.initial_state import (
    InitialStateGenerator,
    get_initial_state_generator,
)
from semble.sequence_generators import SequenceGenerator, get_sequence_generator
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

NUMPY_RNG_SEED = 224691153


def rrmse_u(y_true, y_other):
    error = np.mean(
        np.linalg.norm(y_true - y_other, axis=-1)
        / (np.linalg.norm(y_true) + 1e-9),
        axis=1,
    )
    return error


def rrmse_param(y_true, y_other):
    error = np.linalg.norm(y_true - y_other, axis=-1) / np.linalg.norm(y_true)
    return error


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
        "--time_horizons",
        type=float,
        nargs="+",
        default=[20.0],
    )
    ap.add_argument("--dts", type=float, nargs="+", default=[0.001])

    return ap.parse_args()


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


def output_func(y):
    return jnp.mean(y)


def compute_times_and_errors(
    flumen: Flumen,
    time_horizon: float,
    n_warmup: int,
    dynamics: ParameterisedCellTransmissionModel,
    x0,
    u,
    params,
    delta: float,
    dts: list,
):

    def warmup_and_time(x0, u, params, func):
        ### for dx/du ###
        # y = np.empty((u.shape[0] - n_warmup, u.shape[1], u.shape[2]))

        ### for dx/dparam ###
        y = np.empty((params.shape[0] - n_warmup, params.shape[1]))

        for x_, u_, params_ in zip(
            x0[:n_warmup], u[:n_warmup], params[:n_warmup]
        ):
            func(params_, u_, x_)
        t = time()
        for k, (x_, u_, params_) in enumerate(
            zip(x0[n_warmup:], u[n_warmup:], params[n_warmup:])
        ):
            y[k] = func(params_, u_, x_).block_until_ready()
        return (time() - t) / (x0.shape[0] - n_warmup), y

    def warmup_and_time_batched(x0, u, func):
        func(u[:n_warmup], x0[:n_warmup])
        t = time()
        y = func(u[n_warmup:], x0[n_warmup:])
        return (time() - t) / (x0.shape[0] - n_warmup), y

    dynf_jax = ParameterisedCellTransmissionModel_Jax(dynamics, delta)
    dts.sort()

    ode_term = diffrax.ODETerm(dynf_jax)
    time_vector = np.array([time_horizon])
    ts = diffrax.SaveAt(ts=time_vector)  # type: ignore

    dt = dts[0]

    @equinox.filter_jit
    @equinox.filter_grad
    def diffrax_euler_func(params, u, x):
        y = cast(
            Array,
            diffrax.diffeqsolve(
                ode_term,
                solver=diffrax.Euler(),
                t0=0.0,
                t1=time_horizon,
                dt0=dt,
                y0=x,
                args=(u, params),
                saveat=ts,
                stepsize_controller=diffrax.ConstantStepSize(),
                max_steps=100 + int(time_horizon / dt),
            ).ys,
        )
        return output_func(y)

    t_diffrax_euler, g_true = warmup_and_time(x0, u, params, diffrax_euler_func)

    diffrax_euler_results = []
    for dt in dts[1:]:

        @equinox.filter_jit
        @equinox.filter_grad
        def diffrax_euler_func(params, u, x):
            y = cast(
                Array,
                diffrax.diffeqsolve(
                    ode_term,
                    solver=diffrax.Euler(),
                    t0=0.0,
                    t1=time_horizon,
                    dt0=dt,
                    y0=x,
                    args=(u, params),
                    saveat=ts,
                    stepsize_controller=diffrax.ConstantStepSize(),
                    max_steps=100 + int(time_horizon / dt),
                ).ys,
            )
            return output_func(y)

        t_diffrax_euler, g_diffrax_euler = warmup_and_time(
            x0, u, params, diffrax_euler_func
        )
        error_euler = rrmse_param(g_true, g_diffrax_euler)
        print(f"Euler(dt={dt}): {t_diffrax_euler:.3e} s/traj")
        diffrax_euler_results.append(
            {
                "Method": f"Euler (dt={dt})",
                r"$T$": time_horizon,
                "Time per trajectory (s)": t_diffrax_euler,
                "RRMSE": error_euler,
            }
        )

    @equinox.filter_jit
    @equinox.filter_grad
    def diffrax_tsit5_func(params, u, x):
        y = cast(
            Array,
            diffrax.diffeqsolve(
                ode_term,
                solver=diffrax.Tsit5(),
                t0=0.0,
                t1=time_horizon,
                dt0=dts[0],
                y0=x,
                args=(u, params),
                saveat=ts,
                stepsize_controller=diffrax.PIDController(atol=1e-9, rtol=1e-9),
            ).ys,
        )
        return output_func(y)

    t_diffrax_tsit5, g_diffrax_tsit5 = warmup_and_time(
        x0, u, params, diffrax_tsit5_func
    )
    tsit5_error = rrmse_param(g_true, g_diffrax_tsit5)
    print(f"diffrax(Tsit5): {t_diffrax_tsit5:.3e} s/traj")
    tsit5_results = {
        "Method": "Tsit5",
        r"$T$": time_horizon,
        "Time per trajectory (s)": t_diffrax_tsit5,
        "RRMSE": tsit5_error,
    }

    time_vector = time_vector.reshape((-1, 1))
    flat_model, model_treedef = jax.tree_util.tree_flatten(flumen)

    skips = jnp.floor(time_vector / delta).astype(jnp.uint32)
    tau = ((time_vector - delta * skips) / delta).squeeze()
    skips = skips.squeeze()
    tau_seq = jnp.ones((u.shape[1], 1))
    tau_seq = tau_seq.at[skips, :].set(tau)

    @equinox.filter_jit
    @equinox.filter_grad
    def flumen_func(params, u, x):
        model: Flumen = jax.tree_util.tree_unflatten(model_treedef, flat_model)
        rnn_input = jnp.concatenate((u, tau_seq), axis=-1)
        y = model(x, rnn_input, tau, skips + 1, params)  # type: ignore

        return output_func(y)

    t_flumen, g_flumen = warmup_and_time(x0, u, params, flumen_func)
    print(f"flumen: {t_flumen:.3e} s/traj")
    error_flumen = rrmse_param(g_true, g_flumen)
    flumen_results = {
        "Method": "Flumen",
        r"$T$": time_horizon,
        "Time per trajectory (s)": t_flumen,
        "RRMSE": error_flumen,
    }

    PLOT_GRADIENTS_FLAG = False
    if PLOT_GRADIENTS_FLAG:
        _, axs = plt.subplots(min(8, g_flumen.shape[0]), 1)
        for k, ax in enumerate(axs):
            ax.plot(g_diffrax_tsit5[k + 4], label="Tsit5")
            # ax.plot(g_manual_euler[k+4], label="Euler (diffrax)")
            # ax.plot(g_diffrax_euler[k + 4], label="Euler")
            ax.plot(g_flumen[k + 4], label="flumen")
        axs[0].legend()
        plt.show()

    return *diffrax_euler_results, tsit5_results, flumen_results


def main(args):
    model_path = Path(args.path)
    # import wandb

    # api = wandb.Api()
    # model_artifact = api.artifact(args.path)
    # model_path = Path(model_artifact.download())

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
            n_warmup,
            dynamics,
            x0,
            u,
            params,
            delta,
            args.dts,
        )

        all_results.extend(results)

    times_and_errors = (
        pd.DataFrame(all_results).explode("RRMSE").reset_index(drop=True)
    )
    times_and_errors.to_pickle("grad_timings.pkl")

    # get true times
    times = times_and_errors["Time per trajectory (s)"].unique()

    # jitter times to make it look nicer
    times_and_errors["Time per trajectory (s)"] = times_and_errors[
        "Time per trajectory (s)"
    ].apply(lambda x: x * (1 + uniform(-0.1, 0.1)))

    print(times_and_errors)
    _, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")

    sns.scatterplot(
        times_and_errors,
        x="Time per trajectory (s)",
        y="RRMSE",
        hue="Method",
        palette="colorblind",
        ax=ax,
    )
    for t in times:
        ax.axvline(x=t, alpha=0.2)
    plt.tight_layout()  # helps before saving
    plt.savefig("grad_timings.pdf")
    plt.show()


if __name__ == "__main__":
    main(parse_args())
