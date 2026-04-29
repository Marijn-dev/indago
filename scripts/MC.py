"""Monte carlo simulation of parameter estimation"""

from argparse import ArgumentParser
from pathlib import Path
from indago.estimate import ParameterEstimator
from indago.dataloader import RawNumPyDataset
from indago.model import Dynamics_JAX, DiffraxModel
from flumen_jax import Flumen
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

import re
import wandb
import pickle
import yaml
import jax.numpy as jnp
import numpy as np
import os
import torch

hyperparameters = {
    "n_epochs": 30,  # max number of epochs
    "model": "flumen",  # flumen, diffrax, jax
    "optimizer": "BFGS",  # Adam, GradientDescent, BFGS
    "parameter_loss": "l1_relative",
    "NUMPY_KEY_SEED": 3520758,
}

# only used when model is diffrax
settings_diffrax = {
    "integrator": "Euler",  # Dopri5, Dopri8, Euler, Tsit5
    "dt0": 0.002,  # initial step size
}

# only used when model is jax
settings_jax = {"dt": 0.001}


def parse_args():
    ap = ArgumentParser()

    ap.add_argument("data_path", type=str, help="Path to trajectory dataset")

    ap.add_argument(
        "param_settings",
        type=str,
        help="Path to parameter generator settings file",
    )

    ap.add_argument(
        "runs", type=int, help="Number of parameter estimation runs"
    )

    ap.add_argument(
        "--model_path",
        type=str,
        help="Path to Weights & Biases artifact (required if flumen is used)",
    )

    ap.add_argument("name", type=str, nargs="+", help="Name of the experiment.")

    return ap.parse_args()


def estimation_run(
    init_params, parameter_estimator: ParameterEstimator, epochs: int
):
    est_params = init_params
    for iter in range(epochs):
        est_params, done_estimating = parameter_estimator.train_step(est_params)
        if done_estimating:
            return est_params, iter

    return est_params, iter


def main():
    args = parse_args()
    rng = np.random.default_rng(seed=hyperparameters["NUMPY_KEY_SEED"])
    param_rng = rng.spawn(1)[0]

    timestamp = get_timestamp()
    full_name = "_".join([timestamp] + args.name)
    full_name = re.sub("[^a-zA-Z0-9_-]", "_", full_name)

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
        model: Flumen = return_model(
            hyperparameters["model"],
            None,
            metadata,
            model_path,
            hyperparameters,
        )

    elif hyperparameters["model"] == "diffrax":
        dynamics_jax: Dynamics_JAX = return_dynamics_jax(data["settings"])
        hyperparameters.update(settings_diffrax)
        model: DiffraxModel = return_model(
            hyperparameters["model"], dynamics_jax, None, None, hyperparameters
        )

    elif hyperparameters["model"] == "jax":
        dynamics_jax = return_dynamics_jax(data["settings"])
        hyperparameters.update(settings_jax)
        model = return_model(
            hyperparameters["model"], dynamics_jax, None, None, hyperparameters
        )

    with open(args.param_settings, "r") as f:
        param_settings = yaml.load(f, Loader=yaml.FullLoader)

    hyperparameters["runs"] = args.runs
    hyperparameters["init_param_settings"] = param_settings

    run = wandb.init(
        project="indago (MC)",
        entity="aguiar-kth-royal-institute-of-technology",
        config=hyperparameters,
        name=full_name,
    )

    parameter_generator: ParameterGenerator = get_parameter_generator(
        param_settings["name"], param_settings["args"]
    )

    train_data = RawNumPyDataset(data["train"])
    val_data = RawNumPyDataset(data["val"])
    test_data = RawNumPyDataset(data["test"])

    y, x0, u, t = train_data[0:]
    train_data_args = (
        jnp.array(y),
        jnp.array(x0),
        jnp.array(u),
        jnp.array(t),
        train_data.delta,
        len(train_data),
    )

    optim = get_optimizer(wandb.config["optimizer"])
    true_params = train_data.get_params
    parameter_estimator = ParameterEstimator(
        optim, model, train_data_args, true_params, wandb.config["model"]
    )
    params_loss_fn = get_parameter_loss(wandb.config["parameter_loss"])

    n_succesful_runs = 0
    iterations = []
    param_loss_list = []
    true_params_list = []
    est_params_list = []
    time_list = []
    for i in range(wandb.config["runs"]):
        print("simulation: ", i + 1, "/", wandb.config["runs"])
        time_start = time()
        init_params = parameter_generator.sample(param_rng)
        parameter_estimator.reset(init_params)

        est_params, iter = estimation_run(
            init_params,
            parameter_estimator,
            wandb.config["n_epochs"],
        )

        est_params_list.append(est_params)
        true_params_list.append(true_params)
        est_time = time() - time_start
        print("init params:", init_params, "est params:", est_params)
        print("duration of run:", est_time)

        iterations.append(iter)
        param_loss = params_loss_fn(true_params, est_params)
        param_loss_list.append(param_loss)
        time_list.append(est_time)

        if param_loss < 0.01:
            n_succesful_runs += 1

    results = {
        "iterations": iterations,
        "time_list": time_list,
        "est_params": est_params_list,
        "true_params": true_params_list,
        "param_loss": param_loss_list,
        "n_successul_runs": n_succesful_runs,
        "n_runs": wandb.config["runs"],
    }

    torch.save(results, "results_dict.pth")
    results_artifact = wandb.Artifact(
        name=f"run_data_{wandb.run.id}",
        type="eval_results",
        description="Raw iteration and parameter loss lists",
    )

    results_artifact.add_file("results_dict.pth")
    os.remove("results_dict.pth")
    wandb.log_artifact(results_artifact)

    wandb.log({"images/iterations": log_loss_histogram(iterations, bins=20)})
    # plt.close(fig)
    wandb.log(
        {"images/parameter_loss": log_loss_histogram(param_loss_list, bins=20)}
    )

    wandb.log(
        {
            "succesful_runs": n_succesful_runs,
            "ratio_runs": n_succesful_runs / wandb.config["runs"],
        }
    )

    wandb.summary["succesful_runs"] = n_succesful_runs
    wandb.summary["ratio_runs"] = n_succesful_runs / wandb.config["runs"]


if __name__ == "__main__":
    main()
