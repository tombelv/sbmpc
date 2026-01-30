"""
MPC Controller module - separated from simulation concerns.

This module contains the controller logic independent of simulation.
"""

from functools import partial
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from typing import Union

from sbmpc.model import BaseModel
from sbmpc.solvers import BaseObjective, RolloutGenerator
from sbmpc.sampler import Sampler, MPPISampler
from sbmpc.gains import Gains, MPPIGain
from sbmpc.config import ControllerConfig
from sbmpc.reference import Reference


class MPCController:
    """Model Predictive Control controller.
    
    This class encapsulates the MPC controller logic, separated from
    simulation concerns. It uses a rollout generator to evaluate
    candidate trajectories and a sampler to explore the control space.
    
    Example:
        >>> controller = MPCController.from_config(config, model, objective)
        >>> control_sequence = controller.compute_control(state, reference)
        >>> next_control = control_sequence[0]
    """
    
    def __init__(
        self,
        rollout_generator: RolloutGenerator,
        sampler: Sampler,
        gains: Gains
    ):
        """Initialize controller with components.
        
        Args:
            rollout_generator: Generates and evaluates trajectory rollouts
            sampler: Samples control sequences
            gains: Computes feedback gains (if enabled)
        """
        self.rollout_gen = rollout_generator
        self.sampler = sampler
        self.gains_obj = gains
        self.objective = rollout_generator.objective
        
    @classmethod
    def from_config(
        cls,
        controller_config: ControllerConfig,
        model: BaseModel,
        objective: BaseObjective
    ) -> 'MPCController':
        """Create controller from configuration.
        
        This is the recommended way to create a controller.
        
        Args:
            controller_config: Controller configuration
            model: Dynamics model for rollouts (contains all dimensions)
            objective: Cost function
            
        Returns:
            Configured MPCController instance
        """
        # Create rollout generator (model has all dimensions, no need for model_config)
        rollout_gen = RolloutGenerator(model, objective, controller_config)
        
        # Create sampler (currently only MPPI supported)
        # Now passing model directly so sampler can access actual dimensions
        sampler = MPPISampler(controller_config, model)
        
        # Create gains object
        # Now passing model directly so gains can access actual dimensions
        gains = MPPIGain(controller_config, model)
        
        return cls(rollout_gen, sampler, gains)
    
    def compute_control(
        self,
        state: jnp.ndarray,
        reference: Union[jnp.ndarray, Reference],
        shift_guess: bool = True,
        num_iterations: int = 1
    ) -> jnp.ndarray:
        """Compute optimal control sequence.
        
        Args:
            state: Current state
            reference: Reference trajectory (can be single state or full trajectory)
            shift_guess: Whether to shift previous solution for warm-start
            num_iterations: Number of optimization iterations
            
        Returns:
            Optimal control sequence (horizon x nu)
        """
        optimal_samples = self.sampler.optimal_samples
        gains = self.gains_obj.cur_gains
        
        # Convert reference if needed
        ref_array = reference.as_array() if isinstance(reference, Reference) else reference

        # Run optimization iterations
        for _ in range(num_iterations):
            # Sample perturbations
            samples_delta = self.sampler.sample_input_sequence(self.sampler.master_key)
            
            # Evaluate rollouts
            samples, costs, gradients = self.rollout_gen.do_rollout(
                state, ref_array, optimal_samples, samples_delta, gains
            )
            
            # Update optimal solution
            optimal_samples = self.sampler.update(optimal_samples, samples, costs)
            
            # Update gains if enabled
            self.gains_obj.cur_gains = self.gains_obj.gains_computation(
                costs, samples, gradients
            )
        
        # Prepare for next control call
        if shift_guess:
            self.sampler.optimal_samples = self._shift_guess(optimal_samples)
        else:
            self.sampler.optimal_samples = optimal_samples
            
        return optimal_samples
    
    @partial(jax.jit, static_argnums=(0,))
    def _shift_guess(self, optimal_samples: jnp.ndarray) -> jnp.ndarray:
        """Shift control sequence for warm-start."""
        optimal_samples_shifted = jnp.roll(optimal_samples, shift=-1, axis=0)
        optimal_samples_shifted = optimal_samples_shifted.at[-1, :].set(
            optimal_samples_shifted[-2, :]
        )
        return optimal_samples_shifted
    
    @property
    def gains(self) -> jnp.ndarray:
        """Get current feedback gains."""
        return self.gains_obj.cur_gains
    
    def reset(self):
        """Reset controller state."""
        self.sampler.reset()
        self.gains_obj.reset()


def create_controller(
    controller_config: ControllerConfig,
    model: BaseModel,
    objective: BaseObjective
) -> MPCController:
    """Factory function to create controller.
    
    This is a convenience function that wraps MPCController.from_config.
    
    Args:
        controller_config: Controller configuration
        model: Dynamics model (contains all dimensions)
        objective: Cost function
        
    Returns:
        Configured controller
    """
    return MPCController.from_config(controller_config, model, objective)
