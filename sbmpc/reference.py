"""
Reference data structures for SBMPC.

Use `Reference` to pass structured reference information to controllers
while still exposing a JAX array for computation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import jax.numpy as jnp


@dataclass
class Reference:
    """Base reference container.

    Attributes:
        data: The reference trajectory/state as a JAX array.
        metadata: Optional dictionary for structured access in applications.
    """
    data: jnp.ndarray
    metadata: Optional[Dict[str, Any]] = None

    def as_array(self) -> jnp.ndarray:
        """Return the reference as a JAX array."""
        return self.data
