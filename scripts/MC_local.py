from argparse import ArgumentParser
from pathlib import Path
from indago.estimate import ParameterEstimator
from indago.dataloader import RawNumPyDataset
from indago.model import Dynamics_JAX, DiffraxModel
from flumen_jax import Flumen
from time import time
from semble import (
    get_parameter_generator,
)
from indago.utils import (
    return_model,
    return_dynamics_jax,
    get_optimizer,
    get_parameter_loss,
    get_timestamp,
)

import re
import pickle
import yaml
import jax.numpy as jnp
import numpy as np
import torch

# Key used for rng in init parameter sampling
NUMPY_KEY_SEED = 3520758


def parse_args():
    ap = ArgumentParser()

    ap.add_argument("data_path", type=str, help="Path to trajectory dataset")

    ap.add_argument(
        "method",
        type=str,
        help="Method to use for estimation. Supported: Dopri5, Dopri8, Tsit5, Euler, Flumen",
        default="Flumen",
    )

    ap.add_argument(
        "optimizer",
        type=str,
        help="Optimizer to use for parameter estimation. Supported: GradientDescent, BFGS, Adam",
        default="BFGS",
    )

    ap.add_argument(
        "init_params",
        type=str,
        help="Path to parameter generator settings file, used to sample the initial parameter values.",
    )

    # Optional arguments
    ap.add_argument(
        "--dt",
        type=float,
        help="initial timestep of numerical solver",
        default=0.002,
    )

    ap.add_argument(
        "--n_runs",
        type=int,
        help="number of estimation runs to perform",
        default=50,
    )

    ap.add_argument(
        "--experiment_name", type=str, help="experiment name", default=None
    )

    ap.add_argument(
        "--max_steps",
        type=int,
        help="maximum number of steps in a parameter estimation run",
        default=30,
    )
    return ap.parse_args()


def estimation_run(
    init_params, parameter_estimator: ParameterEstimator, max_steps: int
):
    est_params = init_params
    for step in range(max_steps):
        est_params, estimation_done = parameter_estimator.train_step(est_params)
        if estimation_done:
            return est_params, step

    return est_params, step


def main():
    args = parse_args()
    rng = np.random.default_rng(seed=NUMPY_KEY_SEED)
    param_rng = rng.spawn(1)[0]

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
    dynamics_jax: Dynamics_JAX = return_dynamics_jax(data["settings"])
    dynamics_name = data["settings"]["dynamics"]["name"]
    if dynamics_name == "ParameterisedCellTransmissionModel":
        model_path = Path("models_local_CTM/2704/")
    elif dynamics_name == "VanDerPolParameterised":
        model_path = Path("models_local_vdp/2904/")
    with open(model_path / "metadata.yaml", "r") as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    model: Flumen | DiffraxModel = return_model(
        args.method,
        dynamics_jax,
        metadata,
        model_path,
        args.dt,
    )

    with open(args.init_params, "r") as f:
        init_params_settings = yaml.load(f, Loader=yaml.FullLoader)

    args.init_params = init_params_settings

    parameter_generator = get_parameter_generator(
        init_params_settings["name"], init_params_settings["args"]
    )

    optim = get_optimizer(args.optimizer)
    params_loss_fn = get_parameter_loss("RRMSE")

    true_params = train_data.get_params
    parameter_estimator = ParameterEstimator(
        optim, model, train_data_args, true_params
    )

    n_succesful_runs = 0
    iterations_list = []
    param_loss_list = []
    true_params_list = []
    est_params_list = []
    time_list = []
    for run in range(args.n_runs):
        print("Run: ", run + 1, "/", args.n_runs)
        init_params = parameter_generator.sample(param_rng)
        parameter_estimator.reset(init_params)

        time_start = time()
        est_params, steps = estimation_run(
            init_params,
            parameter_estimator,
            args.max_steps,
        )
        est_time = time() - time_start
        param_loss = params_loss_fn(true_params, est_params)

        print(
            f"Initial params, {init_params}, estimated params: {est_params}, found in {steps + 1} steps and {est_time:.3f} [s]."
        )

        est_params_list.append(est_params)
        true_params_list.append(true_params)
        iterations_list.append(steps)
        param_loss_list.append(param_loss)
        time_list.append(est_time)

        if param_loss < 0.01:
            n_succesful_runs += 1

    results = {
        "iterations": iterations_list,
        "time_list": time_list,
        "est_params": est_params_list,
        "true_params": true_params_list,
        "param_loss": param_loss_list,
        "n_successul_runs": n_succesful_runs,
        "n_runs": args.n_runs,
    }

    # Save and log results to WB
    torch.save(results, "results_dict.pth")


if __name__ == "__main__":
    main()
