import jax
import jax.numpy as jnp
import equinox
import optimistix as optx
from jaxtyping import Float, Array, Bool
from .typing import Parameter, Aux
from flumen_jax import Flumen
from .dataloader import RawNumPyDataset


def L1_relative(params_true: Parameter, params_est: Parameter) -> Float:
    error = jnp.abs(params_est - params_true) / (jnp.abs(params_true) + 1e-8)
    return jnp.mean(error)


def L2_relative(params_true: Parameter, params_est: Parameter) -> Float:
    diff_norm = jnp.linalg.norm(params_est - params_true)
    true_norm = jnp.linalg.norm(params_true)
    return diff_norm / (true_norm + 1e-8)


class ParameterEstimator:
    def __init__(
        self,
        optim: optx.AbstractMinimiser,
        model: Flumen,
        args: tuple[Array, Array, Array, Array, Float, Float],
        init_params: Parameter,
    ):
        self.optim = optim
        self.model = model

        f_struct = jax.ShapeDtypeStruct((), jnp.float32)
        aux_struct = None
        tags = frozenset()

        self.step = equinox.filter_jit(
            equinox.Partial(
                optim.step,
                fn=self._compute_loss,
                args=args,
                options={},
                tags=tags,
            )
        )
        self.terminate = equinox.filter_jit(
            equinox.Partial(
                optim.terminate,
                fn=self._compute_loss,
                args=args,
                options={},
                tags=tags,
            )
        )

        self.state = self.optim.init(
            fn=self._compute_loss,
            y=init_params,
            args=args,
            options={},
            f_struct=f_struct,
            aux_struct=aux_struct,
            tags=tags,
        )

    @equinox.filter_jit
    def _compute_loss(self, params, args) -> tuple[Float, Aux]:
        y, x0, u, t, delta, n_trajectories = args
        skips = jnp.floor(t / delta).astype(jnp.uint32)
        tau = (t - delta * skips) / delta
        eval_trajectory = equinox.filter_vmap(
            self.model.eval_trajectory,
            in_axes=(0, 0, 0, 0, None),
        )
        y_pred = eval_trajectory(x0, u, tau, skips.squeeze(), params)

        loss_val = jnp.mean(jnp.square(y - y_pred))
        # loss_val = jnp.sum(jnp.square(y - y_pred)) / n_trajectories

        aux = None
        return loss_val, aux

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
        return self._compute_loss(params, data_args)[0]
