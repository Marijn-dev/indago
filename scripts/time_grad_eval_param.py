from argparse import ArgumentParser
from math import floor
from pathlib import Path
from time import time
from typing import cast
from indago.model import (
    Diffrax,
    Dynamics_JAX,
    ParameterisedCellTransmissionModelNonSmooth_Jax,
    ParameterisedCellTransmissionModelSmooth_Jax,
    ParameterisedCellTransmissionModel_Numpy,
)
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
from jaxtyping import Array
from scipy.integrate import solve_ivp
from semble.dynamics import ParameterisedCellTransmissionModel, GreenshieldsTraffic, ParameterisedNewellDaganzoTraffic
from semble.parameter_generators import ParameterGenerator,get_parameter_generator

from semble.initial_state import (
    InitialStateGenerator,
    get_initial_state_generator,
)
from semble.sequence_generators import SequenceGenerator, get_sequence_generator

from flumen_jax import Flumen

NUMPY_RNG_SEED = 214690153
### This might explain the parameter estimation taking longer since dy/dtheta might be expensive. Inspect dy/dtheta and look at gradient with time on the x axis for theta=1.92 (the saddle point) in 

class GreenshieldsAdjoint:
    def __init__(
        self, dynamics: GreenshieldsTraffic, delta, time_horizon: float
    ):
        super().__init__()
        self.delta = delta
        self.v0 = dynamics.v0
        self.inv_step = dynamics.inv_step
        self.n = dynamics.n
        self.n_control_differentials = 1 + int(floor(time_horizon / self.delta))

    def __call__(self, t, y, u):
        n_control = jnp.floor(t / self.delta).astype(jnp.uint32)
        u_val = u[n_control, 0]

        rho = y[: self.n]
        r = y[self.n :]

        q_out = self.v0 * rho * (1.0 - rho)
        q0_in = self.v0 * u_val * (1.0 - u_val)
        q_in = jnp.hstack((q0_in, q_out[:-1]))
        drho = self.inv_step * (q_in - q_out)

        p_out = self.v0 * (1.0 - 2.0 * rho)
        p_in = jnp.hstack((0.0, p_out[:-1]))
        r_in = jnp.hstack((0.0, r[:-1]))

        dr = self.inv_step * (
            jnp.tile(p_in, self.n_control_differentials) * r_in
            - jnp.tile(p_out, self.n_control_differentials) * r
        )

        dr = dr.at[n_control * self.n].add(
            self.inv_step * self.v0 * (1.0 - 2.0 * u_val)
        )

        return jnp.concatenate((drho, dr))


class GreenshieldsJax:
    def __init__(self, dynamics: GreenshieldsTraffic, delta):
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
    ap.add_argument("--diffrax_dt0", type=float, default=0.01)

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
    return jnp.sum(y)


def compute_times_and_errors(
    flumen: Flumen,
    time_horizon: float,
    n_warmup: int,
    dynamics: ParameterisedCellTransmissionModel,
    x0,
    u,
    params,
    delta: float,
    dt0=0.01,
):
    def warmup_and_time(x0, u, params, func):
        # for dx/du
        y = np.empty((u.shape[0] - n_warmup, u.shape[1], u.shape[2]))
        # for dx/dparam
        # y = np.empty((params.shape[0] - n_warmup, params.shape[1]))
        for x_, u_, params_ in zip(x0[:n_warmup], u[:n_warmup], params[:n_warmup]):
            func(u_, x_,params_)
        t = time()
        for k, (x_, u_,params_) in enumerate(zip(x0[n_warmup:], u[n_warmup:], params[n_warmup:])):
            y[k] = func(u_, x_,params_).block_until_ready()
        return (time() - t) / (x0.shape[0] - n_warmup), y

    def warmup_and_time_batched(x0, u, func):
        func(u[:n_warmup], x0[:n_warmup])
        t = time()
        y = func(u[n_warmup:], x0[n_warmup:])
        return (time() - t) / (x0.shape[0] - n_warmup), y

    time_vector = np.array([time_horizon])

    dynf = ParameterisedCellTransmissionModelNonSmooth_Jax(dynamics, delta)
    ode_term = diffrax.ODETerm(dynf)
    ts = diffrax.SaveAt(ts=time_vector)  # type: ignore

    @equinox.filter_jit
    @equinox.filter_grad
    def diffrax_euler_func(u, x, params):
        y = cast(
            Array,
            diffrax.diffeqsolve(
                ode_term,
                solver=diffrax.Euler(),
                t0=0.0,
                t1=time_horizon,
                dt0=dt0,
                y0=x,
                args=(u,params),
                saveat=ts,
                stepsize_controller=diffrax.ConstantStepSize(),
                max_steps=100 + int(time_horizon / dt0),
            ).ys,
        )
        return output_func(y)

    t_euler, g_diffrax_euler = warmup_and_time(x0, u, params,diffrax_euler_func)
    print(f"diffrax(Euler): {t_euler:.3e} s/traj")

    @equinox.filter_jit
    @equinox.filter_grad
    def diffrax_tsit5_func(u, x,params):
        y = cast(
            Array,
            diffrax.diffeqsolve(
                ode_term,
                solver=diffrax.Tsit5(),
                t0=0.0,
                t1=time_horizon,
                dt0=dt0,
                y0=x,
                args=(u,params),
                saveat=ts,
                # adjoint=diffrax.BacksolveAdjoint(),
                # stepsize_controller=diffrax.PIDController(atol=1e-6, rtol=1e-3),
                # stepsize_controller=diffrax.ConstantStepSize(),
                stepsize_controller=diffrax.PIDController(atol=1e-12, rtol=1e-12),
            ).ys,
        )
        return output_func(y)

    t_tsit5, g_diffrax_tsit5 = warmup_and_time(x0, u, params,diffrax_tsit5_func)
    print(f"diffrax(Tsit5): {t_tsit5:.3e} s/traj")

    time_vector = time_vector.reshape((-1, 1))
    flat_model, model_treedef = jax.tree_util.tree_flatten(flumen)

    skips = jnp.floor(time_vector / delta).astype(jnp.uint32)
    tau = ((time_vector - delta * skips) / delta).squeeze()
    skips = skips.squeeze()
    tau_seq = jnp.ones((u.shape[1], 1))
    tau_seq = tau_seq.at[skips, :].set(tau)

    @equinox.filter_jit
    @equinox.filter_grad
    def flumen_func(u, x,params):
        model: Flumen = jax.tree_util.tree_unflatten(model_treedef, flat_model)
        rnn_input = jnp.concatenate((u, tau_seq), axis=-1)
        y = model(x, rnn_input, tau, skips + 1, params)  # type: ignore

        return output_func(y)

    t_flumen, g_flumen = warmup_and_time(x0, u, params,flumen_func)
    print(f"flumen: {t_flumen:.3e} s/traj")

    ### This is hard to do with CTM model
    # gs_adj = GreenshieldsAdjoint(dynamics, delta, time_horizon)
    # gs_adj_jit = jax.jit(gs_adj)

    # g_numpy = -1 * np.ones_like(g_flumen)

    # for k, (x, u) in enumerate(zip(x0[n_warmup:], u[n_warmup:])):
    #     init_cond = np.concatenate(
    #         (x, np.zeros(gs_adj.n * gs_adj.n_control_differentials))
    #     )

    #     sol = solve_ivp(
    #         gs_adj_jit,
    #         t_span=(0.0, time_horizon),
    #         y0=init_cond,
    #         method="RK45",
    #         t_eval=time_vector[0],
    #         args=(u,),
    #         atol=1e-9,
    #         rtol=1e-9,
    #     )

    #     adjoints = sol.y[gs_adj.n :, 0]

    #     adjoints = np.stack(
    #         np.split(adjoints, gs_adj.n_control_differentials),
    #         axis=-1,
    #     )

    #     g_numpy[k] = np.sum(adjoints, axis=0, keepdims=True).T

    def error_stats(g_other):
        g_other = g_other.squeeze()
        g_true = g_numpy.squeeze()

        g_true_norm = np.linalg.norm(g_true, axis=-1)
        error = np.linalg.norm(g_true - g_other, axis=-1) / g_true_norm
        mean = np.mean(error, axis=0)
        std = np.std(error, axis=0)

        norm_product = g_true_norm * np.linalg.norm(g_other, axis=-1)
        cossim = np.einsum("ij,ij -> i", g_true, g_other) / norm_product
        mean_cossim = np.mean(cossim, axis=0)

        return mean.item(), std.item(), mean_cossim.item()

    def norm(g_one, g_second):
        g_one_norm = np.linalg.norm(g_one,axis=-1)
        error = np.linalg.norm(g_one-g_second, axis=-1) 
        return np.mean(error, axis=0)

    methods = ("Euler vs Tsit5", "Flumen vs Euler", "Flumen vs Tsit")
    grads = (g_diffrax_euler, g_diffrax_tsit5, g_flumen)
    norm_euler_flumen = norm(g_diffrax_euler,g_flumen)
    norm_euler_tsit = np.mean(norm(g_diffrax_euler,g_diffrax_tsit5))
    norm_euler_euler = norm(g_diffrax_euler,g_diffrax_euler)
    norm_flumen_euler = np.mean(norm(g_flumen,g_diffrax_euler))
    norm_flumen_tsit = np.mean(norm(g_flumen,g_diffrax_tsit5))
    norm_flumen_flumen = norm(g_flumen,g_flumen)
    norm_tsit_flumen = norm(g_diffrax_tsit5,g_flumen)
    norm_tsit_euler = norm(g_diffrax_tsit5,g_diffrax_euler)
    norm_tsit_tsit = norm(g_diffrax_tsit5,g_diffrax_tsit5)

    # print("gradient u shape", g_diffrax_euler.shape)
    # print("norm shape", norm_euler_flumen.shape)
    # # print("Grad flumen:", g_flumen)
    # print("Grad euler:", g_diffrax_euler)
    # print(g_diffrax_euler.shape)
    # if print(np.isnan(g_diffrax_euler).any()):
    #     print("has nans") # True
    # if print(np.isnan(g_diffrax_tsit5).any()):
    #     print("has nans")  # True
    # # print("Grad tsit:", g_diffrax_tsit5)
    # has_nan_per_traj = jnp.isnan(g_diffrax_euler).any(axis=(1, 2))
    # num_traj_with_nan = has_nan_per_traj.sum()
    # print(num_traj_with_nan)
    # has_nan_per_traj = jnp.isnan(g_diffrax_tsit5).any(axis=(1, 2))
    # num_traj_with_nan = has_nan_per_traj.sum()
    # print(num_traj_with_nan)
    # # print(g_diffrax_tsit5)
    # print("max values euler")
    # print(g_diffrax_euler.max(axis=(1,2)))
    times = (t_euler, t_flumen, t_flumen)
    norms = (norm_euler_tsit, norm_flumen_euler, norm_flumen_tsit)
    # results = (
    #     {
    #         "Method": method,
    #         "Time horizon": time_horizon,
    #         "Time per trajectory (s)": time,
    #         "Relative error (mean)": mean_err,
    #         "Relative error (std)": std_err,
    #         "Cosine similarity (mean)": cossim,
    #     }
    #     for method, time, (mean_err, std_err, cossim) in zip(
    #         methods, times, map(error_stats, results)
    #     )
    # )
#     results = (
#     {
#         "Method": method,
#         "Time horizon": time_horizon,
#         "Time per trajectory (s)": time,
#     }
#     for method, time in zip(methods, times)
# )
    results = (
    {
        "Method": method,
        "Time horizon": time_horizon,
        "Time per trajectory (s)": time,
        "Norm": norm,
    }
    for method, time, norm in zip(
        methods, times, norms
    )
)
    # _, axs = plt.subplots(min(8, g_flumen.shape[0]), 1)
    # for k, ax in enumerate(axs):
    #     # ax.plot(g_diffrax_tsit5[k+4], label="Tsit5")
    #     ax.plot(g_diffrax_euler[k+4], label="Euler")
    #     ax.plot(g_flumen[k+4], label="flumen")
    #     # ax.plot(g_numpy[k], "k--", label="Adjoint")
    # axs[0].legend()
    # plt.show()

    return results, grads


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

        results, grads = compute_times_and_errors(
            model,
            time_horizon,
            n_warmup,
            dynamics,
            x0,
            u,
            params,
            delta,
            args.diffrax_dt0,
        )

        all_results.extend(results)

    times_and_errors = pd.DataFrame(all_results)
    grad_euler, grad_tsit5, grad_flumen = grads

    print(times_and_errors)
    _, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")

    ## norm plot
    sns.scatterplot(
        data=times_and_errors,
        x="Time per trajectory (s)",
        y="Norm",
        style="Method",
        hue="Time horizon",
        palette="colorblind",
        ax=ax,
    )

    # sns.scatterplot(
    #     times_and_errors,
    #     x="Time per trajectory (s)",
    #     y="Relative error (mean)",
    #     style="Method",
    #     hue="Time horizon",
    #     ax=ax,
    # )


    # _, ax = plt.subplots()
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # sns.scatterplot(
    #     times_and_errors,
    #     x="Time per trajectory (s)",
    #     y="Cosine similarity (mean)",
    #     style="Method",
    #     hue="Time horizon",
    #     ax=ax,
    # )

    plt.show()


if __name__ == "__main__":
    main(parse_args())
