# from flumen_jax import Flumen
# from jaxtyping import Float, Array, Bool
# from .typing import Parameter, Aux
# from .dataloader import RawNumPyDataset
# from .model import Diffrax

# import jax
# import jax.numpy as jnp
# import equinox as eqx
# import optimistix as optx


# def L1_relative(params_true: Parameter, params_est: Parameter) -> Float:
#     error = jnp.abs(params_est - params_true) / (jnp.abs(params_true) + 1e-8)
#     return jnp.mean(error)


# def L2_relative(params_true: Parameter, params_est: Parameter) -> Float:
#     diff_norm = jnp.linalg.norm(params_est - params_true)
#     true_norm = jnp.linalg.norm(params_true)
#     return diff_norm / (true_norm + 1e-8)


# class ParameterEstimator:
#     def __init__(
#         self,
#         optim: optx.AbstractMinimiser,
#         model: Flumen | Diffrax,
#         args: tuple[Array, Array, Array, Array, Float, Float],
#         init_params: Parameter,
#         model_type,
#     ):
#         self.optim = optim

#         if model_type == "flumen":
#             flat_model, treedef_model = jax.tree_util.tree_flatten(model)
#             model = jax.tree_util.tree_unflatten(treedef_model, flat_model)

#             self.eval_trajectory = eqx.filter_jit(eqx.filter_vmap(
#                 model.eval_trajectory,
#                 in_axes=(0, 0, 0, 0, None),
#             ))
#             self._loss_fn = self._compute_loss_flumen

#         elif model_type == "diffrax":
#             self._loss_fn = self._compute_loss_diffrax
#             self.eval_trajectory = eqx.filter_jit(eqx.filter_vmap(
#             model.eval_trajectory, in_axes=(0, 0, 0, None)
#         ))
#         else:
#             raise ValueError(f"Loss function for {model_type} not supported")

#         f_struct = jax.ShapeDtypeStruct((), jnp.float32)
#         aux_struct = None
#         tags = frozenset()
#         # tags = frozenset({"print"})
#         # options = {"print_every": 1}
#         options = {}
#         self.step = eqx.filter_jit(
#             eqx.Partial(
#                 optim.step,
#                 fn=self._loss_fn,
#                 args=args,
#                 options=options,
#                 tags=tags,
#             ))

#         self.terminate = eqx.filter_jit(
#             eqx.Partial(
#                 optim.terminate,
#                 fn=self._loss_fn,
#                 args=args,
#                 options=options,
#                 tags=tags,
#             ))


#         self.state = self.optim.init(
#             fn=self._loss_fn,
#             y=init_params,
#             args=args,
#             options=options,
#             f_struct=f_struct,
#             aux_struct=aux_struct,
#             tags=tags,
#         )

#     # @eqx.filter_jit
#     def _compute_loss_flumen(self, params, args) -> tuple[Float, Aux]:
#         y, x0, u, t, delta, n_trajectories = args
#         skips = jnp.floor(t / delta).astype(jnp.uint32)
#         tau = (t - delta * skips) / delta


#         y_pred = self.eval_trajectory(x0, u, tau, skips.squeeze(), params)

#         loss_val = jnp.mean(jnp.square(y - y_pred))

#         aux = None
#         return loss_val, aux

#     def _compute_loss_diffrax(self, params, args) -> tuple[Float, Aux]:
#         y, x0, u, t, delta, n_trajectories = args
#         # skips = jnp.floor(t / delta).astype(jnp.uint32)
#         # tau = (t - delta * skips) / delta

#         y_pred = self.eval_trajectory(x0, u, t, params)

#         # integrator can return infinite values and / or NaN values
#         # y_pred = jnp.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

#         loss_val = jnp.mean(jnp.square(y - y_pred))

#         aux = None
#         return loss_val, aux

#     def train_step(self, params: Parameter) -> tuple[Parameter, Bool]:
#         # prev_accepted = self.state.num_accepted_steps
#         # while True:
#         #     params, self.state, aux = self.step(y=params, state=self.state)

#         #     # Check whether this step was accepted
#         #     if self.state.num_accepted_steps > prev_accepted:
#         #         break
#         # self.model = jax.tree_util.tree_unflatten(treedef, flat_model)
#         params, self.state, aux = self.step(y=params, state=self.state)
#         done, result = self.terminate(y=params, state=self.state)
#         return params, done

#     def validate(self, params: Parameter, data: RawNumPyDataset) -> Float:
#         y, x0, u, t = data[0:]
#         data_args = (
#             jnp.array(y),
#             jnp.array(x0),
#             jnp.array(u),
#             jnp.array(t),
#             data.delta,
#             len(data),
#         )
#         return self._loss_fn(params, data_args)[0]
from flumen_jax import Flumen
from jaxtyping import Float, Array, Bool
from .typing import Parameter, Aux
from .dataloader import RawNumPyDataset
from .model import Diffrax

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


class ParameterEstimator:
    def __init__(
        self,
        optim: optx.AbstractMinimiser,
        model: Flumen | Diffrax,
        args: tuple[Array, Array, Array, Array, Float, Float],
        init_params: Parameter,
        model_type,
    ):
        self.optim = optim
        self.model = model
        self.args = args

        if model_type == "flumen":
            self._loss_fn = self._compute_loss_flumen
        elif model_type == "diffrax":
            self._loss_fn = self._compute_loss_diffrax
        else:
            raise ValueError(f"Loss function for {model_type} not supported")

        f_struct = jax.ShapeDtypeStruct((), jnp.float32)
        aux_struct = None
        tags = frozenset()

        self.step = eqx.filter_jit(
            eqx.Partial(
                optim.step,
                fn=self._loss_fn,
                args=self.args,
                options={},
                tags=tags,
            )
        )
        self.terminate = eqx.filter_jit(
            eqx.Partial(
                optim.terminate,
                fn=self._loss_fn,
                args=self.args,
                options={},
                tags=tags,
            )
        )

        self.state = self.optim.init(
            fn=self._loss_fn,
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
            fn=self._loss_fn,
            y=new_init_params,
            args=self.args,
            options={},
            f_struct=f_struct,
            aux_struct=aux_struct,
            tags=tags,
        )

    @eqx.filter_jit
    def _compute_loss_flumen(self, params, args) -> tuple[Float, Aux]:
        y, x0, u, t, delta, n_trajectories = args
        skips = jnp.floor(t / delta).astype(jnp.uint32)
        # print(skips.shape)
        tau = (t - delta * skips) / delta
        # print(tau.shape)
        # print('y', y.shape)
        # print('x0', x0.shape)
        # print('t', t.shape)
        # print(delta.shape)
        # print('skips', skips.shape)
        # print('tau', tau.shape)
        # return

        eval_trajectory = eqx.filter_vmap(
            self.model.eval_trajectory,
            in_axes=(0, 0, 0, 0, None),
        )
        y_pred = eval_trajectory(x0, u, tau, skips.squeeze(-1), params)

        loss_val = jnp.mean(jnp.square(y - y_pred))

        aux = None
        return loss_val, aux

    @eqx.filter_jit
    def _compute_loss_diffrax(self, params, args) -> tuple[Float, Aux]:
        y, x0, u, t, _, n_trajectories = args
        eval_trajectory = eqx.filter_vmap(
            self.model.eval_trajectory, in_axes=(0, 0, 0, None)
        )
        y_pred = eval_trajectory(x0, u, t, params)

        # integrator can return infinite values and / or NaN values
        y_pred = jnp.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        loss_val = jnp.mean(jnp.square(y - y_pred))

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
        return self._loss_fn(params, data_args)[0]
