from flumen_jax import Flumen
from jaxtyping import Float, Array, Bool
from .typing import Parameter, Aux
from .dataloader import RawNumPyDataset
from .model import DiffraxModel, JaxModel

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx


def L1_relative(params_true: Parameter, params_est: Parameter) -> Float:
    error = jnp.abs(params_est - params_true) / (jnp.abs(params_true))
    return jnp.mean(error)


def L2_relative(params_true: Parameter, params_est: Parameter) -> Float:
    diff_norm = jnp.linalg.norm(params_est - params_true)
    true_norm = jnp.linalg.norm(params_true)
    return diff_norm / (true_norm + 1e-8)


def RRMSE_param(y_true, y_other):
    error = jnp.linalg.norm(y_true - y_other, axis=-1) / jnp.linalg.norm(y_true)
    return jnp.mean(error).item()


class ParameterEstimator:
    def __init__(
        self,
        optim: optx.AbstractMinimiser,
        model: Flumen | DiffraxModel | JaxModel,
        args: tuple[Array, Array, Array, Array, Float, Float],
        init_params: Parameter,
    ):
        self.optim = optim
        self.model = model
        self.args = args

        if isinstance(model, Flumen):
            self._train_loss_fn = self._train_loss_flumen
            self._val_loss_fn = self._val_loss_flumen
            self._eval_trajectory = eqx.filter_vmap(
                self.model.eval_trajectory,
                in_axes=(0, 0, 0, 0, None),
            )
        # DiffraxModel | JaxModel
        else:
            self._train_loss_fn = self._train_loss_diffrax
            self._val_loss_fn = self._val_loss_diffrax
            self._eval_trajectory = eqx.filter_vmap(
                self.model.eval_trajectory, in_axes=(0, 0, 0, None)
            )

        f_struct = jax.ShapeDtypeStruct((), jnp.float32)
        aux_struct = None
        tags = frozenset()

        self.step = eqx.filter_jit(
            eqx.Partial(
                optim.step,
                fn=self._train_loss_fn,
                args=self.args,
                options={},
                tags=tags,
            )
        )
        self.terminate = eqx.filter_jit(
            eqx.Partial(
                optim.terminate,
                fn=self._train_loss_fn,
                args=self.args,
                options={},
                tags=tags,
            )
        )

        self.state = self.optim.init(
            fn=self._train_loss_fn,
            y=init_params,
            args=self.args,
            options={},
            f_struct=f_struct,
            aux_struct=aux_struct,
            tags=tags,
        )

    def reset(self, new_init_params: Parameter):
        f_struct = jax.ShapeDtypeStruct((), jnp.float32)
        aux_struct = None
        tags = frozenset()

        self.state = self.optim.init(
            fn=self._train_loss_fn,
            y=new_init_params,
            args=self.args,
            options={},
            f_struct=f_struct,
            aux_struct=aux_struct,
            tags=tags,
        )

    @eqx.filter_jit
    def _train_loss_flumen(self, params, args) -> tuple[Float, Aux]:
        y, x0, u, t, delta, n_trajectories = args
        skips = jnp.floor(t / delta).astype(jnp.uint32)
        tau = (t - delta * skips) / delta

        y_pred = self._eval_trajectory(x0, u, tau, skips.squeeze(-1), params)

        train_loss = jnp.mean(jnp.square(y - y_pred))
        aux = None
        return train_loss, aux

    @eqx.filter_jit
    def _val_loss_flumen(self, params, args) -> tuple[Float, Aux]:
        y, x0, u, t, delta, n_trajectories = args
        skips = jnp.floor(t / delta).astype(jnp.uint32)
        tau = (t - delta * skips) / delta

        y_pred = self._eval_trajectory(x0, u, tau, skips.squeeze(-1), params)
        val_loss = jnp.mean(jnp.sum(jnp.square(y - y_pred), axis=-1))
        return val_loss

    @eqx.filter_jit
    def _train_loss_diffrax(self, params, args) -> tuple[Float, Aux]:
        y, x0, u, t, _, n_trajectories = args

        y_pred = self._eval_trajectory(x0, u, t, params)

        # integrator can return infinite values and / or NaN values
        y_pred = jnp.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        train_loss = jnp.mean(jnp.square(y - y_pred))
        aux = None
        return train_loss, aux

    @eqx.filter_jit
    def _val_loss_diffrax(self, params, args) -> tuple[Float, Aux]:
        y, x0, u, t, _, n_trajectories = args

        y_pred = self._eval_trajectory(x0, u, t, params)

        # integrator can return infinite values and / or NaN values
        y_pred = jnp.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        val_loss = jnp.mean(jnp.sum(jnp.square(y - y_pred), axis=-1))
        return val_loss

    def train_step(self, params: Parameter) -> tuple[Parameter, Bool]:
        params, self.state, aux = self.step(y=params, state=self.state)

        done, result = self.terminate(y=params, state=self.state)
        return params, done

    def validate(self, params: Parameter, data: RawNumPyDataset) -> Float:
        y, x0, u, t = data[0:]
        data_args = (
            jnp.array(y),
            jnp.array(x0),
            jnp.array(u),
            jnp.array(t),
            data.delta,
            len(data),
        )
        return self._val_loss_fn(params, data_args)
