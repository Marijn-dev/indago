import torch
import os
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from indago.estimate import L1_relative, L2_relative, RMSE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import jax.numpy as jnp
from indago.estimate import ParameterEstimator
from indago.dataloader import RawNumPyDataset
from flumen_jax import Flumen
from indago.model import Dynamics_JAX, Diffrax
import yaml
import pickle
from time import time
from semble import (
    get_parameter_generator,
    ParameterGenerator,
)
from indago.utils import (
    return_model,
    return_dynamics_jax,
    get_optimizer,
    get_parameter_loss,
    get_timestamp,
    log_loss_histogram,
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

hyperparameters = {
    "n_epochs": 30,  # max number of epochs
    "model": "flumen",  # flumen or diffrax
    "optimizer": "BFGS",  # Adam, GradientDescent, BFGS
    "parameter_loss": "l1_relative",
    "NUMPY_KEY_SEED": 3520758,
}

# only used when model is diffrax
settings_diffrax = {
    "integrator": "Euler",  # Dopri5, Dopri8, Euler, Tsit5
    "dt0": 0.001,  # initial step size
}


def error_stats(y_true, y_other) -> tuple[float, float]:
    error = np.mean(
        np.linalg.norm(y_true - y_other, axis=-1) / np.linalg.norm(y_true),
        axis=1,
    )

    mean = np.mean(error, axis=0)
    std = np.std(error, axis=0)
    return mean.item(), std.item()


def RRMSE(y_true, y_other):
    rmse = np.sqrt(np.mean((y_true - y_other) ** 2))
    norm = np.sqrt(np.mean(y_true**2))
    return rmse / norm


def parse_args():
    ap = ArgumentParser()
    ap.add_argument(
        "path_flumen",
        type=str,
        help="Path to .pth file "
        "(or, if run with --wandb, path to a Weights & Biases artifact)",
    )
    ap.add_argument(
        "path_diffrax",
        type=str,
        help="Path to .pth file "
        "(or, if run with --wandb, path to a Weights & Biases artifact)",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        help="Path to Weights & Biases artifact (required if flumen is used)",
    )
    ap.add_argument("--data_path", type=str, help="Path to trajectory dataset")

    # ap.add_argument(
    #     "--print_info",
    #     action="store_true",
    #     help="Print training metadata and quit",
    # )
    # ap.add_argument("--continuous_state", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    # ap.add_argument("--time_horizon", type=float, default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    if args.wandb:
        import wandb

        api = wandb.Api()
        model_artifact = api.artifact(args.path_flumen)
        file_path_flumen = Path(model_artifact.download())
        model_artifact = api.artifact(args.path_diffrax)
        file_path_diffrax = Path(model_artifact.download())

        model_run = model_artifact.logged_by()
    else:
        # Standard local path
        file_path = args.path

    with open(file_path_flumen / "results_dict.pth", "rb") as f:
        results_flumen = torch.load(f, weights_only=False, map_location="cpu")

    with open(file_path_diffrax / "results_dict.pth", "rb") as f:
        results_diffrax = torch.load(f, weights_only=False, map_location="cpu")

    est_params_flumen = np.array(results_flumen["est_params"])
    true_params_flumen = np.array(results_flumen["true_params"])
    iterations_flumen = np.array(results_flumen["iterations"])
    times_flumen = np.array(results_flumen["time_list"])
    est_params_diffrax = np.array(results_diffrax["est_params"])
    true_params_diffrax = np.array(results_diffrax["true_params"])
    iterations_diffrax = np.array(results_diffrax["iterations"])
    times_diffrax = np.array(results_diffrax["time_list"])

    plot_RMSE = False
    plot_iterations = False
    plot_times = False
    plot_validation_loss = True

    params_RRMSE_flumen = np.zeros(50)
    params_RRMSE_diffrax = np.zeros(50)
    n_success_runs_flumen = 0
    n_success_runs_diffrax = 0
    for i in range(0, est_params_flumen.shape[0]):
        params_RRMSE_flumen[i] = RRMSE(
            true_params_flumen[i], est_params_flumen[i]
        )
        if params_RRMSE_flumen[i] < 0.01:
            n_success_runs_flumen += 1
        params_RRMSE_diffrax[i] = RRMSE(
            true_params_diffrax[i], est_params_diffrax[i]
        )

        if params_RRMSE_diffrax[i] < 0.01:
            n_success_runs_diffrax += 1
    print(n_success_runs_flumen)
    print(n_success_runs_diffrax)
    if plot_RMSE:
        #### MEDIAN flumen his higher
        #### Mean flumen is lower
        # Create DataFrame in long format
        df = pd.DataFrame(
            {
                "RRMSE": np.concatenate(
                    [params_RRMSE_flumen, params_RRMSE_diffrax]
                ),
                "Method": ["Flumen"] * len(params_RRMSE_flumen)
                + ["Euler"] * len(params_RRMSE_diffrax),
            }
        )

        # Plot
        sns.boxplot(x="Method", y="RRMSE", data=df)

        plt.xlabel("")  # optional (since labels already shown)
        plt.ylabel("RRMSE")
        plt.tight_layout()
        plt.yscale("log")
        plt.savefig("RRMSE_boxplot_MC50_bfgs_Euler.pdf")
        plt.show()
    # if plot_RMSE:
    #     #### MEDIAN flumen his higher
    #     #### Mean flumen is lower
    #     # Create DataFrame in long format
    #     df = pd.DataFrame({
    #         "RMSE": np.concatenate([params_RRMSE_flumen, params_RRMSE_diffrax]),
    #         "Method": ["Flumen"] * len(params_RRMSE_flumen) + ["Diffrax"] * len(params_RRMSE_diffrax)
    #     })

    #     # Plot
    #     sns.boxplot(x="Method", y="RRMSE", data=df)

    #     plt.xlabel("")  # optional (since labels already shown)
    #     plt.ylabel("RMSE")
    #     plt.tight_layout()
    #     plt.yscale("log")
    #     plt.savefig("RRMSE_boxplot_MC50_bfgs_Diffrax.pdf")
    #     plt.show()

    if plot_iterations:
        df = pd.DataFrame(
            {
                "Iterations": np.concatenate(
                    [iterations_flumen, iterations_diffrax]
                ),
                "Method": ["Flumen"] * len(iterations_flumen)
                + ["Euler"] * len(iterations_diffrax),
            }
        )

        # Plot
        sns.boxplot(x="Method", y="Iterations", data=df)

        plt.xlabel("")  # optional (since labels already shown)
        plt.ylabel("Iterations")
        plt.tight_layout()
        # plt.yscale("log")
        plt.savefig("Iterations_boxplot_MC50_bfgs_Euler.pdf")
        plt.show()

    if plot_times:
        df = pd.DataFrame(
            {
                "Duration [s]": np.concatenate([times_flumen, times_diffrax]),
                "Method": ["Flumen"] * len(times_flumen)
                + ["Diffrax"] * len(times_diffrax),
            }
        )

        # Plot
        sns.boxplot(x="Method", y="Duration [s]", data=df)

        plt.xlabel("")  # optional (since labels already shown)
        plt.ylabel("Duration [s]")
        plt.tight_layout()
        plt.yscale("log")
        plt.savefig("Durations_boxplot_MC50_bfgs_Diffrax.pdf")
        plt.show()

    if plot_validation_loss:
        rng = np.random.default_rng(seed=hyperparameters["NUMPY_KEY_SEED"])
        param_rng = rng.spawn(1)[0]

        data_path = Path(args.data_path)
        with data_path.open("rb") as f:
            data = pickle.load(f)

        if hyperparameters["model"] == "flumen":
            assert args.model_path, "no model path given"
            api = wandb.Api()
            model_artifact = api.artifact(args.model_path)
            model_path = Path(model_artifact.download())

            with open(model_path / "metadata.yaml", "r") as f:
                metadata: dict = yaml.load(f, Loader=yaml.FullLoader)
            model_flumen: Flumen = return_model(
                hyperparameters["model"],
                None,
                metadata,
                model_path,
                hyperparameters,
            )

        # hyperparameters["model"] == "diffrax":
        dynamics_jax: Dynamics_JAX = return_dynamics_jax(data["settings"])
        hyperparameters.update(settings_diffrax)
        model_diffrax: Diffrax = return_model(
            "diffrax", dynamics_jax, None, None, hyperparameters
        )
        train_data = RawNumPyDataset(data["train"])
        val_data = RawNumPyDataset(data["val"])
        # test_data = RawNumPyDataset(data["test"])

        y, x0, u, t = val_data[0:]
        train_data_args = (
            jnp.array(y),
            jnp.array(x0),
            jnp.array(u),
            jnp.array(t),
            train_data.delta,
            len(train_data),
        )
        true_params = val_data.get_params
        optim = get_optimizer("BFGS")
        parameter_estimator_flumen = ParameterEstimator(
            optim, model_flumen, train_data_args, true_params, "flumen"
        )

        parameter_estimator_diffrax = ParameterEstimator(
            optim, model_diffrax, train_data_args, true_params, "diffrax"
        )

        traj_loss_RRMSE_flumen = np.zeros(est_params_flumen.shape[0])
        traj_loss_RRMSE_diffrax = np.zeros(est_params_flumen.shape[0])
        for i in range(0, est_params_flumen.shape[0]):
            y_flumen, y_pred_flumen = parameter_estimator_flumen.validate(
                est_params_flumen[i], val_data
            )
            y_diffrax, y_pred_diffrax = parameter_estimator_diffrax.validate(
                est_params_diffrax[i], val_data
            )
            traj_loss_RRMSE_flumen[i] = error_stats(y_flumen, y_pred_flumen)[0]
            traj_loss_RRMSE_diffrax[i] = error_stats(y_diffrax, y_pred_diffrax)[
                0
            ]

        df = pd.DataFrame(
            {
                "RRMSE": np.concatenate(
                    [traj_loss_RRMSE_flumen, traj_loss_RRMSE_diffrax]
                ),
                "Method": ["Flumen"] * len(traj_loss_RRMSE_flumen)
                + ["Diffrax"] * len(traj_loss_RRMSE_diffrax),
            }
        )

        # Plot
        sns.boxplot(x="Method", y="RRMSE", data=df)

        plt.xlabel("")  # optional (since labels already shown)
        plt.ylabel("RRMSE")
        plt.tight_layout()
        plt.yscale("log")
        plt.savefig("vallossRRMSE_boxplot_MC50_bfgs_Diffrax.pdf")
        plt.show()


if __name__ == "__main__":
    main()
