from argparse import ArgumentParser
from pathlib import Path
from indago.estimate import ParameterEstimator
from indago.dataloader import RawNumPyDataset
from time import time
from semble import (
    make_trajectory_sampler,
    TSamplerSpec,
    ParameterisedTrajectorySampler,
)
from indago.utils import (
    return_model,
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
    "n_epochs": 250,
    "model": "flumen",
    "optimizer": "BFGS",
    "parameter_loss": "l1_relative",
}

# only used when model is diffrax
settings_diffrax = {
    "integrator": "Dopri5",
    "dt0": 0.1,  # initial step size
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

    if hyperparameters["model"] == "flumen":
        assert args.model_path, "no model path given"
        api = wandb.Api()
        model_artifact = api.artifact(args.model_path)
        model_path = Path(model_artifact.download())

        with open(model_path / "metadata.yaml", "r") as f:
            metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    elif hyperparameters["model"] == "diffrax":
        model_path = None
        metadata = None
        hyperparameters.update(settings_diffrax)

    run = wandb.init(project="indago", config=hyperparameters, name=full_name)

    with data_path.open("rb") as f:
        data = pickle.load(f)

    sampler_spec: TSamplerSpec = data["settings"]
    sampler: ParameterisedTrajectorySampler = make_trajectory_sampler(
        sampler_spec
    )
    sampler.reset_rngs()
    model = return_model(
        wandb.config["model"], sampler, metadata, model_path, hyperparameters
    )

    train_data = RawNumPyDataset(data["train"])
    val_data = RawNumPyDataset(data["val"])
    test_data = RawNumPyDataset(data["test"])

    delta = train_data.delta

    # Batching is not (yet) implemented
    y, x0, u, t = train_data[0:]
    train_data_args = (
        jnp.array(y),
        jnp.array(x0),
        jnp.array(u),
        jnp.array(t),
        delta,
        len(train_data),
    )

    true_params = train_data.get_params
    init_params = jnp.zeros_like(true_params)  # start estimation with zeros

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
    print_losses(0, train_loss, val_loss, params_loss)

    time_start = time()
    for epoch in range(wandb.config["n_epochs"]):
        est_params, done_training = parameter_estimator.train_step(est_params)
        train_loss = parameter_estimator.validate(est_params, train_data)
        val_loss = parameter_estimator.validate(est_params, val_data)
        params_loss = params_loss_fn(true_params, est_params)
        print_losses(epoch + 1, train_loss, val_loss, params_loss)

        if done_training or early_stop:
            print(f"Stopping training at Epoch {epoch + 1}")
            run.summary["final_train"] = train_loss
            run.summary["final_val"] = val_loss
            run.summary["test"] = parameter_estimator.validate(
                est_params, test_data
            )
            run.summary["est_params"] = est_params
            run.summary["init_params"] = init_params
            run.summary["true_params"] = true_params
            run.summary["params_loss"] = params_loss
            run.summary["training time [s]"] = time() - time_start
            break

        wandb.log(
            {
                "time [s]": time() - time_start,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "params_loss": params_loss,
            }
        )


if __name__ == "__main__":
    main()
