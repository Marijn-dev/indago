from jaxtyping import Array, Float
from typing import TypeVar

Parameter = Float[Array, "parameter_dim"]
Aux = TypeVar("Aux")
