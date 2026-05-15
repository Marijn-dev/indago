from math import floor
from semble import Dynamics
from .typing import State
from jaxtyping import Float
from semble.dynamics import (
    ParameterisedCellTransmissionModel,
    ParameterisedVanDerPol,
)

import diffrax as dfx
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import jax

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


"""Jax implementation of Param CTM model, used for (manual) euler and diffrax integration"""


class ParameterisedCellTransmissionModel_Jax:
    def __init__(self, dynamics: ParameterisedCellTransmissionModel, delta):
        super().__init__()
        self.delta = delta
        self.inv_step = dynamics.dynamics.inv_step
        self.flux = self.flux_interp  # flux_interp / flux_nointerp

    def flux_nointerp(self, x, sigma_i, q_max):
        return jnp.minimum(
            q_max * x / sigma_i,
            (1 - x) * q_max / (1 - sigma_i),
        )

    def flux_interp(self, x, sigma_i, q_max):
        locs = jnp.array([0.0, sigma_i, 1.0])
        vals = jnp.array([0.0, q_max, 0.0])
        return jnp.interp(x, locs, vals)

    def _dx(self, x, u_val, params):
        sigma_i, q_max = params
        x_minus_one = jnp.hstack((u_val, x[:-1]))
        x_plus_one = jnp.hstack((x[1:], jnp.zeros_like(u_val)))

        D_i_minus_one = self.flux(
            jnp.minimum(x_minus_one, sigma_i), sigma_i, q_max
        )
        S_i = self.flux(jnp.maximum(x, sigma_i), sigma_i, q_max)
        q_i_minus_one = jnp.minimum(D_i_minus_one, S_i)

        D_i = self.flux(jnp.minimum(x, sigma_i), sigma_i, q_max)
        S_i_plus_one = self.flux(
            jnp.maximum(x_plus_one, sigma_i), sigma_i, q_max
        )
        q_i = jnp.minimum(D_i, S_i_plus_one)

        dx = self.inv_step * (q_i_minus_one - q_i)

        return dx

    # We use this call for manual euler implementation
    def euler_scan(self, ts, dt, x, u_vals, params):
        us = u_vals[jnp.floor(ts / self.delta).astype(jnp.uint32)]

        def f(x, u):
            return x + dt * self._dx(x, u, params), x

        x_last, xs = jax.lax.scan(f, x, us)
        return jnp.concatenate((xs, jnp.expand_dims(x_last, 0)), axis=0)

    # We use this call for diffrax eq solve
    def __call__(self, t, x, args):
        u, params = args
        index = jnp.floor(t / self.delta).astype(jnp.uint32)
        u_val = u[index]
        return self._dx(x, u_val, params)


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


class ParameterisedVanDerPol_Numpy:
    def __init__(self, dynamics: ParameterisedVanDerPol, delta):
        super().__init__()
        self.dynamics = dynamics
        self.delta = delta

    def _set_parameter(self, parameter):
        self.damping = parameter[0]

    def _dx(self, t, x, u):
        index = int(floor(t / self.delta))
        u_val = u[index]
        p, v = x

        dp = v
        dv = -p + self.damping * (1 - p**2) * v + u_val[0]

        return (dp, dv)

    def __call__(self, t, x, u, params):
        self._set_parameter(params)
        return self._dx(t, x, u)


### Used for parameter estimation with diffrax solvers ###
class DiffraxModel:
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
            stepsize_controller=dfx.ConstantStepSize(),
            max_steps=10000,  # default = 4096
        )

        return solution.ys


### Used for parameter estimation with manual euler ###
class JaxModel:
    def __init__(self, dynamics, dt):
        self._dynamics = dynamics
        self._dt = dt
        time_horizon = 10
        n_steps = 1 + jnp.ceil(time_horizon / self._dt).astype(jnp.uint32)
        self.ts_euler = self._dt * jnp.arange(0.0, n_steps + 1)

    @eqx.filter_jit
    def eval_trajectory(self, x0, u, t_samples, params):
        t_samples = t_samples.reshape(-1)  # [seq_len, 1] -> [seq_len]
        # time_horizon = t_samples[-1]
        ys = self._dynamics.euler_scan(
            self.ts_euler[:-1], self._dt, x0, u, params
        )
        return jax.vmap(
            lambda y: jnp.interp(t_samples, self.ts_euler, y),
            in_axes=1,
            out_axes=1,
        )(ys)
