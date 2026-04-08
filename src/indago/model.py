from semble import Dynamics
from .typing import State
from jaxtyping import Float
from semble.dynamics import ParameterisedCellTransmissionModel

import diffrax as dfx
import jax.numpy as jnp
import equinox as eqx

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


class ParameterisedCellTransmissionModel_Jax(Dynamics_JAX):
    def __init__(self, dynamics: ParameterisedCellTransmissionModel, delta):
        super().__init__(dynamics, delta)
        self.P = dynamics.dynamics.P
        self.delta = delta
        self.inv_step = dynamics.dynamics.inv_step

    def flux(self, x):
        return jnp.interp(x, self.locs, self.vals)

    def _set_parameter(self, parameter):
        locations = parameter[::2]
        values = parameter[1::2]

        idx = jnp.argsort(locations)
        locations = locations[idx]
        values = values[idx]

        self.locs = jnp.zeros(len(locations) + 2)
        self.vals = jnp.zeros(len(values) + 2)

        self.vals = self.vals.at[1:-1].set(values)
        self.locs = self.locs.at[1:-1].set(locations)
        self.locs = self.locs.at[-1].set(self.P)

        self.sigma_i = self.locs[jnp.argmax(self.vals)]

    def _dx(self, t, x, u) -> State:
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

    @eqx.filter_jit
    def eval_trajectory(self, x0, u, t_samples, params):
        t_samples = t_samples.reshape(-1)  # [seq_len, 1] -> [seq_len]
        self.dynamics._set_parameter(params)
        self._ode_term = dfx.ODETerm(self.dynamics._dx)
        solution = dfx.diffeqsolve(
            self._ode_term,
            self._integrator,
            t0=t_samples[0],
            t1=t_samples[-1],
            dt0=self._dt0,
            y0=x0,
            args=u,
            saveat=dfx.SaveAt(ts=t_samples),
            adjoint=dfx.DirectAdjoint(),
            max_steps=10000,  # default = 4096
        )

        return solution.ys
