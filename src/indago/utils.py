from semble import ParameterisedTrajectorySampler
from .estimate import L1_relative, L2_relative
from .model import Diffrax
from flumen_jax import Flumen
from jax import random as jrd

import diffrax as dfx
import optimistix as optx
import equinox as eqx
import datetime


def return_integrator(which: str) -> dfx.AbstractERK:
    if which == "Dopri5":
        return dfx.Dopri5()
    elif which == "Dopri8":
        return dfx.Dopri8()
    else:
        raise ValueError(f"Integrator {which} not supported")


def return_model(
    which: str,
    trajectory_sampler: ParameterisedTrajectorySampler,
    metadata=None,
    model_path=None,
    diffrax_settings=None,
) -> Flumen | Diffrax:
    if which == "flumen":
        model: Flumen = eqx.filter_eval_shape(
            Flumen, **metadata["args"], key=jrd.key(0)
        )

        model: Flumen = eqx.tree_deserialise_leaves(
            model_path / "leaves.eqx", model
        )

    elif which == "diffrax":
        integrator = return_integrator(diffrax_settings["integrator"])
        model = Diffrax(trajectory_sampler, integrator, diffrax_settings["dt0"])

    else:
        raise ValueError(f"Unknown model {which}.")

    return model


def print_header():
    header_msg = (
        f"{'Epoch':>5} :: {'Loss (Train)':>16} :: "
        f"{'Loss (Val)':>16} :: {'Loss (Params)':>16}"
    )

    print(header_msg)
    print("=" * len(header_msg))


def print_losses(
    epoch: int,
    train: float,
    val: float,
    params: float,
):
    print(
        f"{epoch + 1:>5d} :: {train:>16.5e} :: {val:>16.5e} :: {params:>16.5e}"
    )


def get_optimizer(which: str) -> optx.AbstractMinimiser:
    if which == "BFGS":
        return optx.BFGS(
            atol=1e-12, rtol=1e-3
        )  # tolerance has effect on early stopping (and thus total training time)!
    else:
        raise ValueError(f"Unknown optimizer {which}.")


def get_parameter_loss(which: str):
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
