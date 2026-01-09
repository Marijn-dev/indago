from argparse import ArgumentParser
from pathlib import Path
from flumen_jax import Flumen
from jax import random as jrd
from indago.estimate import ParameterEstimator, L1_relative, L2_relative
from indago.dataloader import RawNumPyDataset
from time import time

import datetime
import re
import wandb
import pickle
import yaml
import equinox
import optimistix as optx
import jax.numpy as jnp

# Diffrax model not (yet) implemented
hyperparameters = {
    "n_epochs": 250,
    "model": "flumen",
    "optimizer": "BFGS",
    "parameter_loss": "l1_relative",
}


def parse_args():
    ap = ArgumentParser()

    ap.add_argument("data_path", type=str, help="Path to trajectory dataset")

    ap.add_argument(
        "model_path", type=str, help="Path to Weights & Biases artifact"
    )

    ap.add_argument("name", type=str, nargs="+", help="Name of the experiment.")

    return ap.parse_args()


def get_optimizer(which) -> optx.AbstractMinimiser:
    if which == "BFGS":
        return optx.BFGS(
            atol=1e-12, rtol=1e-3
        )  # tolerance has effect on early stopping (and thus total training time)!
    else:
        raise ValueError(f"Unknown optimizer {which}.")


def get_parameter_loss(which):
    if which == "l1_relative":
        return L1_relative
    elif which == "l2_relative":
        return L2_relative
    else:
        raise ValueError(f"Unknown parameter loss function {which}.")


def get_timestamp() -> str:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    ts = now.strftime("%y%m%d_%H%M")

    return ts


def main():
    args = parse_args()
    data_path = Path(args.data_path)

    timestamp = get_timestamp()
    full_name = "_".join([timestamp] + args.name)
    full_name = re.sub("[^a-zA-Z0-9_-]", "_", full_name)

    api = wandb.Api()
    model_artifact = api.artifact(args.model_path)
    model_path = Path(model_artifact.download())

    with open(model_path / "metadata.yaml", "r") as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    run = wandb.init(project="indago", config=hyperparameters, name=full_name)

    with data_path.open("rb") as f:
        data = pickle.load(f)

    model: Flumen = equinox.filter_eval_shape(
        Flumen, **metadata["args"], key=jrd.key(0)
    )

    model: Flumen = equinox.tree_deserialise_leaves(
        model_path / "leaves.eqx", model
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

    init_params = jnp.array([0.0, 0.0, 0.0])
    true_params = train_data.get_params

    optim = get_optimizer(wandb.config["optimizer"])
    params_loss_fn = get_parameter_loss(wandb.config["parameter_loss"])

    parameter_estimator = ParameterEstimator(
        optim, model, train_data_args, init_params
    )
    est_params, done_training = parameter_estimator.train_step(init_params)
    early_stop = False  # not yet implemented

    time_start = time()
    for epoch in range(wandb.config["n_epochs"]):
        est_params, done_training = parameter_estimator.train_step(est_params)

        train_loss = parameter_estimator.validate(est_params, train_data)
        val_loss = parameter_estimator.validate(est_params, val_data)
        params_loss = params_loss_fn(true_params, est_params)

        if done_training or early_stop:
            print(f"Stopping training at Epoch {epoch}")
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
