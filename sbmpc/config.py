"""
Clean configuration module for SBMPC.

This module provides dataclass-based configuration that separates:
- Model configuration (state dimensions, actuator limits, dynamics)
- Controller configuration (horizon, samples, gains, device/dtype)
- Simulation configuration (time step, iterations, visualization)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np

MODEL_PARAMETRIC_INTEGRATOR_TYPES = ["si_euler", "euler", "rk4", "custom_discrete"]


class DynamicsModel(Enum):
    """Type of dynamics model to use."""
    CUSTOM = "custom"
    MJX = "mjx"


class Solver(Enum):
    """Type of MPC solver."""
    MPPI = "mppi"


@dataclass
class ModelConfig:
    """Configuration for the model being controlled.

    Attributes:
        dynamics_model: Type of dynamics model to use
        dynamics_fn: Custom dynamics function (required for CUSTOM model)
        nq: Number of position coordinates
        nv: Number of velocity coordinates
        nu: Number of control inputs
        input_min: Minimum control input limits (shape: nu,)
        input_max: Maximum control input limits (shape: nu,)
        q_init: Initial position configuration (shape: nq,)
        scene_path: Optional path to MuJoCo scene XML file
        mjx_kinematic: Whether to use kinematic mode for MJX
        integrator_type: Integrator type for parametric models [si_euler, euler, rk4, custom_discrete]
    """
    dynamics_model: DynamicsModel
    dynamics_fn: Optional[Callable] = None
    nq: int = 0
    nv: int = 0
    nu: int = 0
    np: int = 0
    input_min: Optional[jnp.ndarray] = None
    input_max: Optional[jnp.ndarray] = None
    q_init: Optional[jnp.ndarray] = None
    scene_path: Optional[str] = None
    mjx_kinematic: bool = False
    integrator_type: str = "si_euler"
    
    @property
    def nx(self) -> int:
        """Total state dimension."""
        return self.nq + self.nv
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dynamics_model == DynamicsModel.CUSTOM and self.dynamics_fn is None:
            raise ValueError("dynamics_fn is required for CUSTOM dynamics model")
        if self.dynamics_model == DynamicsModel.MJX and self.scene_path is None:
            raise ValueError("scene_path is required for MJX dynamics model")
        
        # For CUSTOM models, validate dimensions and input bounds
        if self.dynamics_model == DynamicsModel.CUSTOM:
            if self.input_min is not None and len(self.input_min) != self.nu:
                raise ValueError(f"input_min length {len(self.input_min)} != nu {self.nu}")
            if self.input_max is not None and len(self.input_max) != self.nu:
                raise ValueError(f"input_max length {len(self.input_max)} != nu {self.nu}")
            if self.q_init is not None and len(self.q_init) != self.nq:
                raise ValueError(f"q_init length {len(self.q_init)} != nq {self.nq}")


@dataclass
class ControllerConfig:
    """Configuration for the MPC controller.

    Attributes:
        dt: Time step for control (seconds)
        horizon: Planning horizon (number of steps)
        num_samples: Number of trajectories to sample
        lambda_: Temperature parameter for MPPI
        std_dev: Standard deviation for sampling (shape: nu,)
        initial_guess: Initial control guess (shape: nu,)
        num_control_points: Number of spline control points (if using smoothing)
        smoothing: Type of smoothing ("Spline" or None)
        use_gains: Whether to compute feedback gains
        use_sensitivity: Whether to use sensitivity information
        solver_type: Type of solver to use
        device: JAX device to use
        dtype: Data type for computations
    """
    dt: float
    horizon: int
    num_samples: int = 1000
    lambda_inv: float = 1.0
    std_dev: Optional[jnp.ndarray] = None
    initial_guess: Optional[jnp.ndarray] = None
    num_control_points: int = 0
    smoothing: Optional[str] = None
    use_gains: bool = False
    use_sensitivity: bool = False
    solver_type: Solver = Solver.MPPI
    device: jax.Device = field(default_factory=lambda: jax.devices()[0])
    dtype: type = jnp.float32
    
    def __post_init__(self):
        """Validate configuration."""
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if self.std_dev is None:
            raise ValueError("std_dev must be provided for the controller")
        if self.smoothing is not None and self.smoothing not in ["Spline"]:
            raise ValueError(f"smoothing must be 'Spline' or None, got {self.smoothing}")
        if self.num_control_points < 0:
            raise ValueError(f"num_control_points must be non-negative, got {self.num_control_points}")


@dataclass
class SimulationConfig:
    """Configuration for simulation.

    Attributes:
        dt: Simulation time step (seconds)
        num_iterations: Number of simulation steps
        visualize: Whether to enable visualization
    """
    dt: float
    num_iterations: int
    visualize: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {self.num_iterations}")


