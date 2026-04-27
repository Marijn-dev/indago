from argparse import ArgumentParser
from pathlib import Path
from indago.estimate import ParameterEstimator
from indago.dataloader import RawNumPyDataset
from indago.model import Dynamics_JAX
from time import time
from flumen_jax import Flumen
from indago.utils import (
    return_model,
    return_dynamics_jax,
    get_optimizer,
    get_parameter_loss,
    get_timestamp,
    print_header,
    print_losses,
)

import re
import wandb
import pickle
import yaml
import jax.numpy as jnp

hyperparameters = {
    "n_epochs": 500,
    "model": "diffrax",  # flumen or diffrax
    "optimizer": "BFGS",  # Adam, GradientDescent, BFGS
    "parameter_loss": "RRMSE",  # l1_relative, l2_relative, RRMSE
    # "initial_parameter": [0.0814, 0.2669],  # dimension is data model dependent
    "initial_parameter": [0.0],  # dimension is data model dependent
}

# only used when model is diffrax
settings_diffrax = {
    "integrator": "Tsit5",  # Dopri5, Dopri8, Euler, Tsit5
    "dt0": 0.01,  # initial step size
}


def parse_args():
    ap = ArgumentParser()

    ap.add_argument("data_path", type=str, help="Path to trajectory dataset")

    ap.add_argument(
        "--model_path",
        type=str,
        help="Path to Weights & Biases artifact (required if flumen is used)",
    )

    ap.add_argument("name", type=str, nargs="+", help="Name of the experiment.")

    return ap.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)

    timestamp = get_timestamp()
    full_name = "_".join([timestamp] + args.name)
    full_name = re.sub("[^a-zA-Z0-9_-]", "_", full_name)

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
        model = return_model(
            hyperparameters["model"], dynamics_jax, None, None, hyperparameters
        )

    run = wandb.init(
        project="indago",
        entity="aguiar-kth-royal-institute-of-technology",
        config=hyperparameters,
        name=full_name,
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

    true_params = train_data.get_params
    init_params = jnp.array(
        wandb.config["initial_parameter"], dtype=jnp.float32
    )
    print("Starting estimation with: ", init_params)
    print("True params: ", true_params)

    optim = get_optimizer(wandb.config["optimizer"])
    params_loss_fn = get_parameter_loss(wandb.config["parameter_loss"])

    parameter_estimator = ParameterEstimator(
        optim, model, train_data_args, init_params, wandb.config["model"]
    )
    est_params, done_training = parameter_estimator.train_step(init_params)
    early_stop = False  # not yet implemented

    train_loss = parameter_estimator.validate(est_params, train_data)
    val_loss = parameter_estimator.validate(est_params, val_data)
    params_loss = params_loss_fn(true_params, est_params)

    print_header()
    print_losses(0, train_loss, val_loss, params_loss, est_params)

    wandb.log(
        {
            # "time [s]": 0,
            "epoch": 0,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "params_loss": params_loss,
            "params_est": est_params[0],
        }
    )
    print("starting estimation...")
    time_start = time()
    for epoch in range(wandb.config["n_epochs"]):
        est_params, done_training = parameter_estimator.train_step(est_params)
        train_loss = parameter_estimator.validate(est_params, train_data)
        val_loss = parameter_estimator.validate(est_params, val_data)
        params_loss = params_loss_fn(true_params, est_params)
        print_losses(epoch + 1, train_loss, val_loss, params_loss, est_params)
        est_time = time() - time_start
        wandb.log(
            {
                "time [s]": est_time,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "params_loss": params_loss,
                "params_est": est_params[0],
            }
        )
        if done_training or early_stop:
            print(
                f"Stopping training at Epoch {epoch + 1}, estimation took {est_time:.3f} [s]"
            )
            param_found = True if params_loss < 0.05 else False
            print(
                f"estimation done in {epoch + 1} iterations, parameter found: {param_found}"
            )
            print(f"final parameter error {params_loss}")
            run.summary["final_train"] = train_loss
            run.summary["final_val"] = val_loss
            run.summary["test"] = parameter_estimator.validate(
                est_params, test_data
            )
            run.summary["est_params"] = est_params
            run.summary["init_params"] = init_params
            run.summary["true_params"] = true_params
            run.summary["params_loss"] = params_loss
            run.summary["training time [s]"] = est_time
            run.summary["iterations"] = epoch + 1
            break


if __name__ == "__main__":
    main()
