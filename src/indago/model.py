from semble import ParameterisedTrajectorySampler
from .typing import State
from jaxtyping import Float

import diffrax as dfx
import jax.numpy as jnp


class Diffrax:
    def __init__(
        self,
        trajectory_sampler: ParameterisedTrajectorySampler,
        integrator: dfx.AbstractERK,
        dt0: Float,
    ):
        self._trajectory_sampler = trajectory_sampler
        self._ode_term = dfx.ODETerm(self._f)
        self._integrator = integrator
        self.initial_time = 0
        self._dt0 = dt0  # significantly affects training time

    def _f(self, t, x, args) -> State:
        # vector field in jax-friendly notation
        u, params = args

        # fix -> if this isn't here it doesn't learn, prob cause params has to be part of args (?)
        self._trajectory_sampler._dyn.gen_parameter(None, params)

        n_control = jnp.array(
            (t - self.initial_time) / self._trajectory_sampler._delta, dtype=int
        )
        u_val = u[n_control]
        return jnp.stack([*self._trajectory_sampler._dyn(x, u_val)])

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
            adjoint=dfx.DirectAdjoint(),
            # max_steps=10000, # default = 4096
        )

        return solution.ys
