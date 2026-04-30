from pathlib import Path
import seaborn as sns
import pandas as pd
import diffrax
import torch
import pickle
import os
import equinox
import jax
import jax.numpy as jnp
import jax.random as jrd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from flumen_jax import Flumen
from semble.dynamics import (
    ParameterisedCellTransmissionModel,
)

from indago.dataloader import RawNumPyDataset

from indago.model import (
    ParameterisedCellTransmissionModel_Jax,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "font.size": 18,
    }
)


def rrmse_param(y_true, y_other):
    # y_true: (n_trajectories, state)
    error = np.linalg.norm(y_true - y_other, axis=-1) / np.linalg.norm(y_true)

    return error


def rrmse_traj(y_true, y_other):
    # y_true: (n_trajectories, time, state)
    error = np.mean(
        np.linalg.norm(y_true - y_other, axis=-1) / np.linalg.norm(y_true),
        axis=1,
    )
    mean = np.mean(error, axis=0)
    return mean.item()


def solve_flumen_traj(flat_model, model_treedef, t, x0, u, delta, params):
    model: Flumen = jax.tree_util.tree_unflatten(model_treedef, flat_model)
    skips = jnp.floor(t / delta).astype(jnp.uint32)
    tau = (t - delta * skips) / delta

    return model.eval_trajectory(x0, u, tau, skips.squeeze(), params)


def rrmse_trajectory(
    flumen,
    dynamics,
    time_horizon,
    delta,
    data,
    est_params_flumen,
    est_params_diffrax,
):

    def compute_trajectories(x0, u, est_param, time, y, func):
        for k, (x_, u_, time_) in enumerate(zip(x0, u, time)):
            y[k] = func(x_, u_, est_param, time_)

    test_data = RawNumPyDataset(data["test"])
    y_true, x0, u, time = test_data[0:]
    y_true = np.array(y_true)
    x0 = np.array(x0)
    u = np.array(u)
    t = np.array(time)
    y_diffrax = np.empty_like(y_true)
    y_flumen = np.empty_like(y_true)

    dynf_jax = ParameterisedCellTransmissionModel_Jax(dynamics, delta)
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()
    ode_term = diffrax.ODETerm(dynf_jax)
    dt = 0.002

    @jax.jit
    def euler_func(x, u, params, time_vector):
        t_samples = time_vector.reshape(-1)  # [seq_len, 1] -> [seq_len]
        return diffrax.diffeqsolve(
            ode_term,
            solver,
            t0=0.0,
            t1=time_horizon,
            dt0=dt,
            y0=x,
            args=(u, params),
            saveat=diffrax.SaveAt(ts=t_samples),
            stepsize_controller=stepsize_controller,
            max_steps=100 + int(time_horizon / dt),
        ).ys

    compute_trajectories(x0, u, est_params_diffrax, t, y_diffrax, euler_func)
    traj_RRMSE_diffrax = rrmse_traj(y_true, y_diffrax)

    flat_model, model_treedef = jax.tree_util.tree_flatten(flumen)

    solve_flumen = solve_flumen_traj

    @jax.jit
    def flumen_func(x, u, params, time_vector):
        time_vector = time_vector.reshape((-1, 1))
        return solve_flumen(
            flat_model,
            model_treedef,
            time_vector,
            x,
            u,
            delta,
            params,
        )

    compute_trajectories(x0, u, est_params_flumen, t, y_flumen, flumen_func)
    traj_RRMSE_flumen = rrmse_traj(y_true, y_flumen)

    return traj_RRMSE_diffrax, traj_RRMSE_flumen


def main():

    # Flumen model used for MC
    model_path = "models_local_CTM/2704/"
    model_path = Path(model_path)

    with open(model_path / "metadata.yaml", "r") as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    like_model = equinox.filter_eval_shape(
        Flumen, **metadata["args"], key=jrd.key(0)
    )
    model: Flumen = equinox.tree_deserialise_leaves(
        model_path / "leaves.eqx", like_model
    )

    # data used for MC
    data_path = Path("data/pCTM_T10_N100_s033_q028.pkl")
    data_path = Path(data_path)
    with data_path.open("rb") as f:
        data = pickle.load(f)

    ds = metadata["data_settings"]
    dynamics = ParameterisedCellTransmissionModel(**ds["dynamics"]["args"])

    delta = data["settings"]["control_delta"]
    time_horizon = data["args"]["time_horizon"]

    # Results from MC run
    flumen_run = torch.load("MC/ctm/flumen/results_dict.pth")
    diffrax_run = torch.load("MC/ctm/diffrax/results_dict.pth")

    ### Computational performance ###
    iterations_flumen = np.array(flumen_run["iterations"])
    iterations_diffrax = np.array(diffrax_run["iterations"])
    times_flumen = np.array(flumen_run["time_list"])
    times_diffrax = np.array(diffrax_run["time_list"])

    ### Estimation accuracy ###
    true_params = np.array(flumen_run["true_params"])
    est_params_flumen = np.array(flumen_run["est_params"])
    est_params_diffrax = np.array(diffrax_run["est_params"])
    params_RRMSE_flumen = rrmse_param(true_params, est_params_flumen)
    params_RRMSE_diffrax = rrmse_param(true_params, est_params_diffrax)
    threshold = 0.001

    flumen_below = np.sum(params_RRMSE_flumen < threshold)
    diffrax_below = np.sum(params_RRMSE_diffrax < threshold)

    print(
        f"Flumen parameter RRMSE below {threshold}:",
        flumen_below / true_params.shape[0] * 100,
        "%",
    )
    print(
        f"Euler parameter RRMSE below {threshold}:",
        diffrax_below / true_params.shape[0] * 100,
        "%",
    )

    # RRMSE loss over unseen trajectories, per estimated parameter
    traj_RRMSE_flumen_list = []
    traj_RRMSE_diffrax_list = []
    for est_param_flumen, est_param_diffrax in zip(
        est_params_flumen, est_params_diffrax
    ):
        traj_RRMSE_diffrax, traj_RRMSE_flumen = rrmse_trajectory(
            model,
            dynamics,
            time_horizon,
            delta,
            data,
            est_param_flumen,
            est_param_diffrax,
        )
        traj_RRMSE_flumen_list.append(traj_RRMSE_flumen)
        traj_RRMSE_diffrax_list.append(traj_RRMSE_diffrax)

    traj_RRMSE_flumen_np = np.array(traj_RRMSE_flumen_list)
    traj_RRMSE_diffrax_np = np.array(traj_RRMSE_diffrax_list)

    ### Plotting ###
    plots = [
        ("steps", iterations_flumen, iterations_diffrax, "Steps", True),
        ("time", times_flumen, times_diffrax, "Time (s)", True),
        (
            "parameter",
            params_RRMSE_flumen,
            params_RRMSE_diffrax,
            "Parameter RRMSE",
            True,
        ),
    ]

    save_dir = "ctm_MC_results"
    os.makedirs(save_dir, exist_ok=True)  # creates folder if it
    plots = [
        ("steps", iterations_flumen, iterations_diffrax, "Steps", False),
        ("duration", times_flumen, times_diffrax, "Duration (s)", True),
        (
            "parameter_RRMSE",
            params_RRMSE_flumen,
            params_RRMSE_diffrax,
            "Parameter RRMSE",
            True,
        ),
        (
            "trajectory_RRMSE",
            traj_RRMSE_flumen_np,
            traj_RRMSE_diffrax_np,
            "Trajectory RRMSE",
            True,
        ),
    ]
    for name, fl, di, ylabel, logscale in plots:
        df = pd.DataFrame(
            {
                "Value": np.concatenate([fl, di]),
                "Method": (["Flumen"] * len(fl)) + (["Euler"] * len(di)),
            }
        )

        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(x="Method", y="Value", data=df, width=0.6)

        if logscale:
            ax.set_yscale("log")

        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=20)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name}.pdf")
        plt.close()


if __name__ == "__main__":
    main()
