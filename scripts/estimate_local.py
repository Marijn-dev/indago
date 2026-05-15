from argparse import ArgumentParser
from pathlib import Path
from time import time
from indago.estimate import ParameterEstimator
from indago.dataloader import RawNumPyDataset
from indago.model import Dynamics_JAX, DiffraxModel
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

import os
import re
import pickle
import yaml
import jax.numpy as jnp


def parse_args():
    ap = ArgumentParser()

    ap.add_argument("data_path", type=str, help="Path to trajectory dataset")

    ap.add_argument(
        "method",
        type=str,
        help="Method to use for estimation. Supported: Dopri5, Dopri8, Tsit5, Euler, [model_path], where model_path is the path to a Flumen model",
    )

    ap.add_argument(
        "optimizer",
        type=str,
        help="Optimizer to use for parameter estimation. Supported: GradientDescent, BFGS, Adam",
        default="GradientDescent",
    )

    ap.add_argument(
        "init_params",
        type=float,
        nargs="+",
        default=None,
        help="Initial parameter value",
    )

    # Optional arguments
    ap.add_argument(
        "--dt",
        type=float,
        help="initial timestep of numerical solver",
        default=0.01,
    )

    ap.add_argument(
        "--experiment_name", type=str, help="experiment name", default=None
    )

    ap.add_argument(
        "--max_steps",
        type=int,
        help="maximum number of steps in parameter estimation",
        default=500,
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Create experiment name
    timestamp = get_timestamp()
    parts = [timestamp]
    if args.experiment_name is not None:
        parts.append(args.experiment_name)
    full_name = "_".join(parts)
    full_name = re.sub("[^a-zA-Z0-9_-]", "_", full_name)

    # Load in data
    data_path = Path(args.data_path)
    with data_path.open("rb") as f:
        data = pickle.load(f)

    train_data = RawNumPyDataset(data["train"])
    val_data = RawNumPyDataset(data["val"])

    y, x0, u, t = train_data[0:]
    train_data_args = (
        jnp.array(y),
        jnp.array(x0),
        jnp.array(u),
        jnp.array(t),
        train_data.delta,
        len(train_data),
    )

    # Load in and create appropriate model
    if args.method in ["Dopri5", "Dopri8", "Tsit5", "Euler"]:
        method_name = args.method
        dynamics_jax: Dynamics_JAX = return_dynamics_jax(data["settings"])
        model: DiffraxModel = return_model(
            args.method,
            dynamics_jax,
            None,
            None,
            args.dt,
        )
    else:
        method_name = "Flumen"
        model_path = Path(args.method)
        with open(model_path / "metadata.yaml", "r") as f:
            metadata: dict = yaml.load(f, Loader=yaml.FullLoader)
            model: Flumen = return_model(
                method_name,
                None,
                metadata,
                model_path,
                None,
            )

    # save directory
    dynamics_name = data["settings"]["dynamics"]["name"]
    if dynamics_name == "VanDerPolParameterised":
        dyn_name = "vdp"
    elif dynamics_name == "ParameterisedCellTransmissionModel":
        dyn_name = "ctm"
    else:
        dyn_name = dynamics_name
    save_dir = f"results/estimation/{dyn_name}/{method_name}"

    optim = get_optimizer(args.optimizer)
    params_loss_fn = get_parameter_loss("RRMSE")

    true_params = train_data.get_params
    init_params = jnp.array(args.init_params, dtype=jnp.float32)

    parameter_estimator = ParameterEstimator(
        optim, model, train_data_args, init_params
    )

    print("Initial parameters: ", init_params)
    print("True parameter: ", true_params)
    est_params, estimation_done = parameter_estimator.train_step(init_params)

    # Used to initialize optimization state
    train_loss = parameter_estimator.validate(est_params, train_data)
    val_loss = parameter_estimator.validate(est_params, val_data)
    params_loss = params_loss_fn(true_params, est_params)

    print_header()
    print_losses(0, train_loss, val_loss, params_loss, est_params)

    val_loss_list = []
    est_params_list = []
    params_loss_list = []
    time_start = time()
    for step in range(args.max_steps):
        est_params, estimation_done = parameter_estimator.train_step(est_params)
        train_loss = parameter_estimator.validate(est_params, train_data)
        val_loss = parameter_estimator.validate(est_params, val_data)
        params_loss = params_loss_fn(true_params, est_params)
        print_losses(step + 1, train_loss, val_loss, params_loss, est_params)
        est_time = time() - time_start

        val_loss_list.append(val_loss)
        est_params_list.append(est_params)
        params_loss_list.append(params_loss)

        if estimation_done:
            break

    print(
        f"Estimated params: {est_params}, found in {step + 1} steps and {est_time:.3f} [s]."
    )

    results_dict = {
        "val_losses": val_loss_list,
        "est_params": est_params_list,
        "params_loss": params_loss_list,
        "method": args.method,
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results_dict.pkl"), "wb") as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    main()
