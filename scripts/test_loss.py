from pathlib import Path
from flumen_jax import Flumen
from argparse import ArgumentParser
from indago.dataloader import TestNumPyDataset

import pickle
import equinox
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import yaml


def rrmse(y_true, y_other):
    # y_true: (n_trajectories, time, state)
    error = np.sqrt(
        np.mean(np.sum((y_true - y_other) ** 2, axis=-1), axis=-1)
        / np.mean(np.sum(y_true**2, axis=-1), axis=-1)
    )
    mean = np.mean(error, axis=0)
    return mean.item()


def solve_flumen_batch(flat_model, model_treedef, t, x0, u, delta, params):
    model: Flumen = jax.tree_util.tree_unflatten(model_treedef, flat_model)
    skips = jnp.floor(t / delta).astype(jnp.uint32)
    tau = (t - delta * skips) / delta
    skips = skips.squeeze()

    def eval(x0_, u_, tau_, skip, params_):
        tau_seq = jnp.ones((u_.shape[0], 1))
        tau_seq = tau_seq.at[skip, :].set(tau_)
        rnn_input = jnp.concatenate((u_, tau_seq), axis=-1)

        return model(x0_, rnn_input, tau_, skip + 1, params_)

    return jax.vmap(eval, in_axes=(None, None, 0, 0, None))(
        x0, u, tau, skips, params
    )


def main(args):

    def compute_trajectories(x0, u, params, time, y, func):
        for k, (x_, u_, params_, time_) in enumerate(zip(x0, u, params, time)):
            y[k] = func(x_, u_, params_, time_)

    data_path = Path(args.data_path)
    data_path = Path(data_path)
    with data_path.open("rb") as f:
        data = pickle.load(f)
    delta = data["settings"]["control_delta"]

    if args.wandb:
        import wandb

        api = wandb.Api()
        model_artifact = api.artifact(args.model_path)
        model_path = Path(model_artifact.download())

    else:
        model_path = Path(args.model_path)

    with open(model_path / "metadata.yaml", "r") as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)
    like_model = equinox.filter_eval_shape(
        Flumen, **metadata["args"], key=jrd.key(0)
    )
    model: Flumen = equinox.tree_deserialise_leaves(
        model_path / "leaves.eqx", like_model
    )

    test_data = TestNumPyDataset(data["test"])
    y_true, x0, u, time, params = test_data[0:]
    y_true = np.array(y_true)
    x0 = np.array(x0)
    u = np.array(u)
    t = np.array(time)
    y_flumen = np.empty_like(y_true)

    flat_model, model_treedef = jax.tree_util.tree_flatten(model)

    solve_flumen = solve_flumen_batch

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

    compute_trajectories(x0, u, params, t, y_flumen, flumen_func)
    rrmse_test = rrmse(y_true, y_flumen)
    print(f"RRMSE over test {test_data.n_traj} trajectories: {rrmse_test}")


def parse_args():
    ap = ArgumentParser()

    ap.add_argument(
        "data_path",
        type=str,
        help="Path to data folder",
    )

    ap.add_argument(
        "model_path",
        type=str,
        help="Path to model folder",
    )
    ap.add_argument(
        "--wandb",
        action="store_true",
        help="use if model is wandb",
    )

    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
