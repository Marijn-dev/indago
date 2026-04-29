from semble import Dynamics, get_dynamics
from .estimate import L1_relative, L2_relative, RRMSE
from .model import (
    DiffraxModel,
    Dynamics_JAX,
    ParameterisedCellTransmissionModel_Jax,
    JaxModel,
)
from flumen_jax import Flumen
from jax import random as jrd

import diffrax as dfx
import optimistix as optx
import equinox as eqx
import datetime
import optax
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb


def return_dynamics_jax(data_settings):
    which = data_settings["dynamics"]["name"]
    dynamics: Dynamics = get_dynamics(which, data_settings["dynamics"]["args"])
    delta = data_settings["control_delta"]

    if which == "VanDerPolParameterised":
        return Dynamics_JAX(dynamics, delta)
    elif which == "ParameterisedCellTransmissionModel":
        # return ParameterisedCellTransmissionModelNonSmooth_Jax(dynamics, delta)
        return ParameterisedCellTransmissionModel_Jax(dynamics, delta)
    else:
        print(f"Data model {which} not implemented")


def return_integrator(which: str) -> dfx.AbstractERK:
    if which == "Dopri5":
        return dfx.Dopri5()
    elif which == "Dopri8":
        return dfx.Dopri8()
    elif which == "Euler":
        return dfx.Euler()
    elif which == "Tsit5":
        return dfx.Tsit5()
    else:
        raise ValueError(f"Integrator {which} not supported")


def return_model(
    which: str,
    dynamics_jax: Dynamics_JAX,
    metadata=None,
    model_path=None,
    settings=None,
) -> Flumen | DiffraxModel | JaxModel:
    if which == "flumen":
        model: Flumen = eqx.filter_eval_shape(
            Flumen, **metadata["args"], key=jrd.key(0)
        )

        model: Flumen = eqx.tree_deserialise_leaves(
            model_path / "leaves.eqx", model
        )

    elif which == "diffrax":
        integrator = return_integrator(settings["integrator"])
        model = DiffraxModel(dynamics_jax, integrator, settings["dt0"])

    elif which == "jax":
        model = JaxModel(dynamics_jax, settings["dt"])

    else:
        raise ValueError(f"Unknown model {which}.")

    return model


def print_header():
    header_msg = (
        f"{'Epoch':>5} :: {'Loss (Train)':>16} :: "
        f"{'Loss (Val)':>16} :: {'Loss (Params)':>16} :: {'Est Params'}"
    )

    print(header_msg)
    print("=" * len(header_msg))


def print_losses(
    epoch: int,
    train: float,
    val: float,
    params: float,
    est_params: float,
):
    print(
        f"{epoch:>5d} :: {train:>16.5e} :: {val:>16.5e} :: {params:>16.5e} :: {est_params[0]}"
    )


def get_optimizer(which: str) -> optx.AbstractMinimiser:
    if which == "BFGS":
        return optx.BFGS(
            atol=1e-12, rtol=1e-3
        )  # tolerance has effect on early stopping (and thus total training time)!
    elif which == "GradientDescent":
        return optx.GradientDescent(atol=1e-12, rtol=1e-3, learning_rate=2e-2)
    elif which == "Adam":
        return optx.OptaxMinimiser(
            optax.adam(learning_rate=1e-3), atol=1e-12, rtol=1e-3
        )
    else:
        raise ValueError(f"Unknown optimizer {which}.")


def get_parameter_loss(which: str):
    if which == "l1_relative":
        return L1_relative
    elif which == "l2_relative":
        return L2_relative
    elif which == "RRMSE":
        return RRMSE
    else:
        raise ValueError(f"Unknown parameter loss function {which}.")


def get_timestamp() -> str:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    ts = now.strftime("%y%m%d_%H%M")

    return ts


def log_loss_histogram(loss_list, title="", bins=20):
    """
    Plots a histogram of the losses and logs it to wandb.

    Args:
        loss_list (list or np.array): list of loss values
        title (str): plot title
        bins (int): number of histogram bins
    """
    loss_array = np.array(loss_list)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(loss_array, bins=bins, stat="density", kde=True, ax=ax)

    # ax.set_xlabel("Loss")
    ax.set_ylabel("Probability Density")
    ax.set_title(title)

    image = wandb.Image(fig)
    plt.close(fig)

    return image

    # Clear the figure to avoid overlaps in subsequent calls


# if __name__ == "__main__":
#     losses = [0.1, 0.2, 0.3, 0.5, 0.3, 0.6, 0.7, 0.2, 0.4]
#     log_loss_histogram(losses)
