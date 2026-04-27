from semble import Dynamics
from .typing import State
from jaxtyping import Float
from semble.dynamics import (
    ParameterisedCellTransmissionModel,
    ParameterisedNewellDaganzoTraffic,
)
from math import floor

# from numpy.typing import Arraylike, NDArray
import diffrax as dfx
import jax.numpy as jnp
import equinox as eqx
import numpy as np


"""Diffrax (Jax/JIT friendly) implementation of semble dynamics"""


class Dynamics_JAX:
    def __init__(self, dynamics: Dynamics, delta):
        self.delta = delta
        self._dynamics = dynamics

    def _set_parameter(self, params):
        self._dynamics._set_parameter(None, params)

    def _dx(self, t, x, u) -> State:
        index = jnp.floor(t / self.delta).astype(jnp.uint32)
        u_val = u[index]
        return jnp.stack([*self._dynamics(x, u_val)])

    def __call__(self, t, x, args):
        u, params = args
        self._set_parameter(params)
        return self._dx(t, x, u)


"""Jax (Jax/JIT friendly) implementation of Param CTM model (smooth)"""


class ParameterisedCellTransmissionModelSmooth_Jax(Dynamics_JAX):
    def __init__(self, dynamics: ParameterisedCellTransmissionModel, delta):
        super().__init__(dynamics, delta)
        self.delta = delta
        self.inv_step = dynamics.dynamics.inv_step
        self.locs = None
        self.vals = None
        self.sigma_i = None
        self.flux = self.flux_nointerp  # flux_interp / flux_nointerp

    def flux_nointerp(self, x):
        return jnp.minimum(
            self.q_max * x / self.sigma_i,
            (1 - x) * self.q_max / (1 - self.sigma_i),
        )

    def flux_interp(self, x):
        locs = jnp.array([0.0, self.sigma_i, 1.0])
        vals = jnp.array([0.0, self.q_max, 0.0])
        return jnp.interp(x, locs, vals)

    def _set_parameter(self, parameter):
        self.sigma_i = parameter[0]
        self.q_max = parameter[1]

    def a(self, y1, y2, gamma=10):
        return jnp.exp(gamma * y1) / (jnp.exp(gamma * y1) + jnp.exp(gamma * y2))

    def softmax(self, y_1, y_2):
        a_y = self.a(y_1, y_2)
        return a_y * y_1 + (1 - a_y) * y_2

    def softmin(self, y_1, y_2):
        a_y = self.a(-y_1, -y_2)
        return a_y * y_1 + (1 - a_y) * y_2

    def _dx(self, t, x, u):
        index = jnp.floor(t / self.delta).astype(jnp.uint32)
        u_val = u[index]
        x_minus_one = jnp.roll(x, 1).at[0].set(u_val[0])
        x_plus_one = jnp.roll(x, -1).at[-1].set(0.0)

        D_i_minus_one = self.flux(self.softmin(x_minus_one, self.sigma_i))
        S_i = self.flux(self.softmax(x, self.sigma_i))
        q_i_minus_one = self.softmin(D_i_minus_one, S_i)

        D_i = self.flux(self.softmin(x, self.sigma_i))
        S_i_plus_one = self.flux(self.softmax(x_plus_one, self.sigma_i))
        q_i = self.softmin(D_i, S_i_plus_one)

        dx = self.inv_step * (q_i_minus_one - q_i)
        return dx

    def __call__(self, t, x, args):
        u, params = args
        self._set_parameter(params)
        return self._dx(t, x, u)


"""Jax (Jax/JIT friendly) implementation of Param CTM model (nonsmooth)"""


class ParameterisedCellTransmissionModelNonSmooth_Jax(Dynamics_JAX):
    def __init__(self, dynamics: ParameterisedCellTransmissionModel, delta):
        super().__init__(dynamics, delta)
        self.delta = delta
        self.inv_step = dynamics.dynamics.inv_step
        self.locs = None
        self.vals = None
        self.sigma_i = None
        self.flux = self.flux_interp  # flux_interp / flux_nointerp

    def flux_nointerp(self, x):
        return jnp.minimum(
            self.q_max * x / self.sigma_i,
            (1 - x) * self.q_max / (1 - self.sigma_i),
        )

    def flux_interp(self, x):
        locs = jnp.array([0.0, self.sigma_i, 1.0])
        vals = jnp.array([0.0, self.q_max, 0.0])
        return jnp.interp(x, locs, vals)

    def _set_parameter(self, parameter):
        self.sigma_i = parameter[0]
        self.q_max = parameter[1]

    def _dx(self, t, x, u):
        index = jnp.floor(t / self.delta).astype(jnp.uint32)
        u_val = u[index]
        x_minus_one = jnp.roll(x, 1).at[0].set(u_val[0])
        x_plus_one = jnp.roll(x, -1).at[-1].set(0.0)

        D_i_minus_one = self.flux(jnp.minimum(x_minus_one, self.sigma_i))
        S_i = self.flux(jnp.maximum(x, self.sigma_i))
        q_i_minus_one = jnp.minimum(D_i_minus_one, S_i)

        D_i = self.flux(jnp.minimum(x, self.sigma_i))
        S_i_plus_one = self.flux(jnp.maximum(x_plus_one, self.sigma_i))
        q_i = jnp.minimum(D_i, S_i_plus_one)

        dx = self.inv_step * (q_i_minus_one - q_i)
        return dx

    def __call__(self, t, x, args):
        u, params = args
        self._set_parameter(params)
        return self._dx(t, x, u)


"""Numpy implementation of Param CTM model (nonsmooth)"""


class ParameterisedCellTransmissionModel_Numpy:
    def __init__(self, dynamics: ParameterisedCellTransmissionModel, delta):
        super().__init__()
        self.delta = delta
        self.inv_step = dynamics.dynamics.inv_step
        self.locs = None
        self.vals = None
        self.sigma_i = None
        self.flux = self.flux_interp  # flux_interp / flux_nointerp

    def flux_nointerp(self, x):
        return np.minimum(
            self.q_max * x / self.sigma_i,
            (1 - x) * self.q_max / (1 - self.sigma_i),
        )

    def flux_interp(self, x):
        locs = np.array([0.0, self.sigma_i, 1.0])
        vals = np.array([0.0, self.q_max, 0.0])
        return np.interp(x, locs, vals)

    def _set_parameter(self, parameter):
        self.sigma_i = parameter[0]
        self.q_max = parameter[1]

    def a(self, y1, y2, gamma=10):
        return np.exp(gamma * y1) / (np.exp(gamma * y1) + np.exp(gamma * y2))

    def softmax(self, y_1, y_2):
        a_y = self.a(y_1, y_2)
        return a_y * y_1 + (1 - a_y) * y_2

    def softmin(self, y_1, y_2):
        a_y = self.a(-y_1, -y_2)
        return a_y * y_1 + (1 - a_y) * y_2

    def _dx(self, t, x, u):
        index = int(floor(t / self.delta))
        u_val = u[index]
        x_minus_one = np.roll(x, 1)
        x_minus_one[0] = u_val[0]
        x_plus_one = np.roll(x, -1)
        x_plus_one[-1] = 0.0

        D_i_minus_one = self.flux(np.minimum(x_minus_one, self.sigma_i))
        S_i = self.flux(np.maximum(x, self.sigma_i))
        q_i_minus_one = np.minimum(D_i_minus_one, S_i)

        D_i = self.flux(np.minimum(x, self.sigma_i))
        S_i_plus_one = self.flux(np.maximum(x_plus_one, self.sigma_i))
        q_i = np.minimum(D_i, S_i_plus_one)

        dx = self.inv_step * (q_i_minus_one - q_i)
        return dx

    def __call__(self, t, y, u, params):
        self._set_parameter(params)
        return self._dx(t, y, u)


class Diffrax:
    def __init__(
        self,
        dynamics: Dynamics_JAX,
        integrator: dfx.AbstractERK,
        dt0: Float,
    ):
        self.dynamics = dynamics
        self._integrator = integrator
        self.initial_time = 0
        self._dt0 = dt0
        self._ode_term = dfx.ODETerm(self.dynamics)

    # this filter jit is necessary
    @eqx.filter_jit
    def eval_trajectory(self, x0, u, t_samples, params):
        t_samples = t_samples.reshape(-1)  # [seq_len, 1] -> [seq_len]
        solution = dfx.diffeqsolve(
            self._ode_term,
            self._integrator,
            t0=t_samples[0],
            t1=t_samples[-1],
            dt0=self._dt0,
            y0=x0,
            args=(u, params),
            saveat=dfx.SaveAt(ts=t_samples),
            # adjoint=dfx.DirectAdjoint(),
            # adjoint=dfx.BacksolveAdjoint(),
            # adjoint=dfx.RecursiveCheckpointAdjoint(),
            # stepsize_controller=dfx.PIDController(atol=1e-3, rtol=1e-6),
            stepsize_controller=dfx.ConstantStepSize(),
            # stepsize_controller=dfx.PIDController(atol=1e-2, rtol=1e-2),
            max_steps=1000000000,  # default = 4096
        )
        # print(solution.adjoint)

        return solution.ys
