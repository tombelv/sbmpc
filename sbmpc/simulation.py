"""
Simulation runner module - clean separation from controller logic.

This module handles simulation orchestration, visualization, and data collection.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import time
import numpy as np
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from sbmpc.model import BaseModel, ModelMjx, create_model
from sbmpc.controller import MPCController
from sbmpc.config import SimulationConfig, ModelConfig, DynamicsModel
from sbmpc.reference import Reference


class MujocoVisualizer:
    """Wrapper for MuJoCo visualization."""
    
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """Initialize visualizer.
        
        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data object
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        
        # Launch persistent viewer
        self.viewer = mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=True,
                                                   show_right_ui=False)
    
    def set_qpos(self, qpos: np.ndarray):
        """Update model position for visualization.
        
        Args:
            qpos: Position configuration array
        """
        self.mj_data.qpos = np.array(qpos, dtype=np.float64)
        mujoco.mj_fwdPosition(self.mj_model, self.mj_data)
        self.viewer.sync()
    
    def is_running(self) -> bool:
        """Check if visualization window is open."""
        return self.viewer.is_running()
    
    def close(self):
        """Close visualization."""
        if self.viewer.is_running():
            self.viewer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False



class SimulationRunner:
    """Runs MPC simulations with optional visualization.
    
    This class handles the simulation loop, keeping the controller
    logic separate. It collects trajectory data and optionally
    visualizes the system.
    
    Example:
        >>> runner = SimulationRunner(config, model, controller, initial_state, reference)
        >>> runner.run()
        >>> states, controls = runner.get_trajectory()
    """
    
    def __init__(
        self,
        simulation_config: SimulationConfig,
        model: BaseModel,
        controller: MPCController,
        initial_state,
        reference: Reference,
        visualizer: Optional['Visualizer'] = None
    ):
        """Initialize simulation runner.
        
        Args:
            simulation_config: Simulation configuration
            model: Dynamics model for simulation
            controller: MPC controller
            initial_state: Initial state (array or MuJoCo data)
            reference: Reference object
            visualizer: Optional visualizer
        """
        self.simulation_config = simulation_config
        self.model = model
        self.controller = controller
        self.reference = reference
        self.visualizer = visualizer
        
        # Simulation state
        self.current_state = initial_state
        self.iter = 0
        
        # Extract state vector function based on initial state type
        if isinstance(initial_state, (np.ndarray, jnp.ndarray)):
            self.get_state_vec = lambda: self.current_state
        elif isinstance(initial_state, (mjx.Data, mujoco.MjData)):
            if model.kinematic:
                self.get_state_vec = lambda: jnp.array(self.current_state.qpos)
            else:
                self.get_state_vec = lambda: jnp.concatenate([
                    self.current_state.qpos, 
                    self.current_state.qvel
                ])
        else:
            raise ValueError("Invalid initial state type")
        
        # Initialize trajectory storage
        num_steps = simulation_config.num_iterations
        state_dim = self.get_state_vec().size
        
        self.state_trajectory = np.zeros((num_steps + 1, state_dim))
        self.control_trajectory = np.zeros((num_steps, model.nu))
        self.state_trajectory[0] = self.get_state_vec()
        
        self.dt = simulation_config.dt
        
    def run(self):
        """Run the simulation loop."""
        num_steps = self.simulation_config.num_iterations
        
        if self.visualizer is not None:
            self._run_with_visualization()
        else:
            self._run_headless()
            
    def _run_headless(self):
        """Run simulation without visualization."""
        num_steps = self.simulation_config.num_iterations
        
        while self.iter < num_steps:
            self._step()
            
    def _run_with_visualization(self):
        """Run simulation with visualization using MuJoCo viewer."""
        num_steps = self.simulation_config.num_iterations
        
        try:
            if not hasattr(self.model, 'mj_model') or self.model.mj_model is None:
                print("Warning: Visualization not available for this model")
                self._run_headless()
                return
            
            while self.visualizer.is_running() and self.iter < num_steps:
                step_start = time.time()
                
                # Step simulation
                self._step()
                
                # Update visualization with current state
                state_qpos = self.get_state_vec()[:self.model.nq]
                self.visualizer.set_qpos(state_qpos)
                
                # Sleep to maintain real-time if needed
                elapsed = time.time() - step_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
            
            # Close visualizer when done
            self.visualizer.close()
                
        except Exception as e:
            if self.visualizer is not None:
                self.visualizer.close()
            print(f"Visualization error: {e}")
            raise e
            
    def _step(self):
        """Execute one simulation step."""
        # Compute control
        time_start = time.time_ns()
        current_state_vec = self.get_state_vec()
        
        print(f"Iteration: {self.iter}")
        print(f"Current state: {current_state_vec}")
        
        control_sequence = self.controller.compute_control(
            current_state_vec,
            self.reference,
            num_iterations=1
        ).block_until_ready()
        
        control = control_sequence[0].block_until_ready()
        
        comp_time_ms = 1e-6 * (time.time_ns() - time_start)
        print(f"Computation time: {comp_time_ms:.3f} ms")
        
        # Store control
        self.control_trajectory[self.iter] = control
        
        # Simulate dynamics
        self.current_state = self.model.integrate_sim(
            self.current_state,
            control,
            self.dt
        )
        
        # Store state
        self.iter += 1
        self.state_trajectory[self.iter] = self.get_state_vec()
        
    def get_trajectory(self):
        """Get recorded trajectory.
        
        Returns:
            Tuple of (state_trajectory, control_trajectory)
        """
        return self.state_trajectory[:self.iter+1], self.control_trajectory[:self.iter]
    
    def get_final_state(self):
        """Get final state."""
        return self.state_trajectory[self.iter]





def create_simulation(
    model: BaseModel,
    controller: MPCController,
    simulation_config: SimulationConfig,
    reference: Reference,
) -> SimulationRunner:
    """Create a simulation runner from pre-created model and controller.
    
    Args:
        model: Dynamics model for simulation
        controller: Controller instance to use
        simulation_config: Simulation configuration
        reference: Reference object
        
    Returns:
        Configured SimulationRunner ready to run
        
    Example:
        >>> model, initial_state = create_model(model_config)
        >>> controller = create_controller(controller_config, model, objective)
        >>> sim = create_simulation(model, controller, simulation_config, reference)
        >>> sim.run()
        >>> states, controls = sim.get_trajectory()
    """
    initial_state = model.initial_state
    
    # Create visualizer if requested in config
    visualizer = None
    if simulation_config.visualize:
        # Visualizer works for MJX models or custom models with scene_path
        if hasattr(model, 'mj_model') and model.mj_model is not None:
            visualizer = MujocoVisualizer(model.mj_model, model.mj_data)
        else:
            print("Warning: Visualization requested but model has no MuJoCo scene (scene_path not provided)")
    
    runner = SimulationRunner(
        simulation_config,
        model,
        controller,
        initial_state,
        reference,
        visualizer
    )
    
    # Warm up JIT compilation using state vector (not mjx.Data)
    state_vec = runner.get_state_vec()
    _ = controller.compute_control(state_vec, reference, shift_guess=False).block_until_ready()
    
    return runner
