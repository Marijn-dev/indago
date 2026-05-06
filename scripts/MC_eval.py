from pathlib import Path
from argparse import ArgumentParser
from flumen_jax import Flumen
from indago.dataloader import RawNumPyDataset
from indago.utils import return_dynamics_jax, return_model

import seaborn as sns
import pandas as pd
import pickle
import os
import equinox
import jax
import jax.numpy as jnp
import jax.random as jrd
import matplotlib.pyplot as plt
import numpy as np
import yaml


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
    error = np.linalg.norm(y_true - y_other, axis=-1) / np.linalg.norm(
        y_true, axis=-1
    )

    return error


def rrmse_traj(y_true, y_other):
    # y_true: (n_trajectories, time, state)
    #
    error = np.sqrt(
        np.mean(np.sum((y_true - y_other) ** 2, axis=-1), axis=-1)
        / np.mean(np.sum(y_true**2, axis=-1), axis=-1)
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
    dynamics_jax,
    delta,
    data,
    est_params_flumen,
    est_params_other,
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
    y_other = np.empty_like(y_true)
    y_flumen = np.empty_like(y_true)

    @jax.jit
    def other_func(x, u, params, time_vector):
        return dynamics_jax.eval_trajectory(x, u, time_vector, params)

    compute_trajectories(x0, u, est_params_other, t, y_other, other_func)
    traj_RRMSE_other = rrmse_traj(y_true, y_other)

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

    return traj_RRMSE_other, traj_RRMSE_flumen


def parse_args():

    ap = ArgumentParser()

    ap.add_argument("data_path", type=str, help="Path to data used for MC")

    ap.add_argument("results_flumen", type=str, help="Path to pkl MC results")

    ap.add_argument("results_other", type=str, help="Path to pkl MC results")

    return ap.parse_args()


def main():

    args = parse_args()

    # Load in data and results
    data_path = Path(args.data_path)
    with data_path.open("rb") as f:
        data = pickle.load(f)
    data_path = Path(args.results_flumen)
    with data_path.open("rb") as f:
        results_flumen = pickle.load(f)
    data_path = Path(args.results_other)
    with data_path.open("rb") as f:
        results_other = pickle.load(f)

    ds = data["settings"]
    dynamics_name = ds["dynamics"]["name"]
    if dynamics_name == "ParameterisedCellTransmissionModel":
        model_path = Path("models/ctm/")
        save_dir = "results/MC/ctm"
    elif dynamics_name == "VanDerPolParameterised":
        model_path = Path("models/vdp/")
        save_dir = "results/MC/vdp"

    # load in Flumen and numerical solver
    with open(model_path / "metadata.yaml", "r") as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)
    like_model = equinox.filter_eval_shape(
        Flumen, **metadata["args"], key=jrd.key(0)
    )
    model: Flumen = equinox.tree_deserialise_leaves(
        model_path / "leaves.eqx", like_model
    )
    dynamics_jax = return_dynamics_jax(ds)
    jax_model = return_model(
        results_other["method"], dynamics_jax, None, None, results_other["dt"]
    )

    delta = ds["control_delta"]

    ### Computational performance ###
    iterations_flumen = np.array(results_flumen["steps"])
    iterations_other = np.array(results_other["steps"])
    times_flumen = np.array(results_flumen["time_list"])
    times_other = np.array(results_other["time_list"])

    ### Estimation accuracy ###
    true_params = np.array(results_flumen["true_params"])
    est_params_flumen = np.array(results_flumen["est_params"])
    est_params_other = np.array(results_other["est_params"])
    params_RRMSE_flumen = rrmse_param(true_params, est_params_flumen)
    params_RRMSE_other = rrmse_param(true_params, est_params_other)
    threshold = 0.01  # threshold when classified as successful run

    flumen_below = np.sum(params_RRMSE_flumen < threshold)
    other_below = np.sum(params_RRMSE_other < threshold)

    other_method = results_other["method"]
    print(
        f"Flumen parameter RRMSE below {threshold}:",
        flumen_below / true_params.shape[0] * 100,
        "% of runs",
    )
    print(
        f"{other_method} parameter RRMSE below {threshold}:",
        other_below / true_params.shape[0] * 100,
        "% of runs",
    )

    # RRMSE loss over unseen trajectories, per estimated parameter
    traj_RRMSE_flumen_list = []
    traj_RRMSE_other_list = []
    for est_param_flumen, est_param_other in zip(
        est_params_flumen, est_params_other
    ):
        traj_RRMSE_other, traj_RRMSE_flumen = rrmse_trajectory(
            model,
            jax_model,
            delta,
            data,
            est_param_flumen,
            est_param_other,
        )
        traj_RRMSE_flumen_list.append(traj_RRMSE_flumen)
        traj_RRMSE_other_list.append(traj_RRMSE_other)

    traj_RRMSE_flumen_np = np.array(traj_RRMSE_flumen_list)
    traj_RRMSE_other_np = np.array(traj_RRMSE_other_list)

    ### Plotting ###
    os.makedirs(save_dir, exist_ok=True)  # creates folder if it
    plots = [
        ("steps", iterations_flumen, iterations_other, "Steps", False),
        ("duration", times_flumen, times_other, "Duration (s)", True),
        (
            "parameter_RRMSE",
            params_RRMSE_flumen,
            params_RRMSE_other,
            "Relative error (parameter)",
            True,
        ),
        (
            "trajectory_RRMSE",
            traj_RRMSE_flumen_np,
            traj_RRMSE_other_np,
            "Relative error (trajectory)",
            True,
        ),
    ]

    for name, fl, di, ylabel, logscale in plots:
        df = pd.DataFrame(
            {
                "Value": np.concatenate([fl, di]),
                "Method": (["Flumen"] * len(fl))
                + ([f"{other_method}"] * len(di)),
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
