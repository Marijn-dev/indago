from .typing import Parameter
from flumen import ParamaterisedRawTrajectoryDataset
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp


class RawNumPyDataset:
    init_state: Float[Array, "dlen state_dim"]
    time: Float[Array, "dlen seq_len 1"]
    state: Float[Array, "dlen seq_len output_dim"]
    control_seq: Float[Array, "dlen seq_len control_dim"]
    parameter: Float[Array, "dlen parameter_dim"]

    state_dim: int
    output_dim: int
    control_dim: int
    parameter_dim: int

    def __init__(self, data: ParamaterisedRawTrajectoryDataset):
        (
            self.state,
            self.init_state,
            self.control_seq,
            self.time,
            self.parameter,
        ) = jax.tree_map(
            jnp.asarray,
            (
                data.state,
                data.init_state,
                data.control_seq,
                data.time,
                data.parameter,
            ),
        )

        self.n_traj = data.n_traj
        self.delta = data.delta
        self.state_dim = data.state_dim
        self.control_dim = data.control_dim
        self.output_dim = data.output_dim
        self.parameter_dim = data.parameter_dim

    def __getitem__(self, index) -> tuple[Array, Array, Array, Array]:
        return (
            self.state[index],
            self.init_state[index],
            self.control_seq[index],
            self.time[index],
        )

    def __len__(self) -> Float:
        return self.n_traj

    @property
    def get_params(self) -> Parameter:
        assert jnp.array_equal(self.parameter[0], self.parameter[1]), (
            "parameters are not the same across trajectories"
        )
        return self.parameter[0]
