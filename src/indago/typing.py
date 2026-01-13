from jaxtyping import Array, Float
from typing import TypeVar

Parameter = Float[Array, "parameter_dim"]
State = Float[Array, "state_dim"]
Aux = TypeVar("Aux")
