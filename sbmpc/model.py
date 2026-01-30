from typing import Optional, Dict

import numpy as np
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import mujoco
from mujoco import mjx

from sbmpc.config import ModelConfig, DynamicsModel, MODEL_PARAMETRIC_INTEGRATOR_TYPES


class BaseModel(ABC):
    def __init__(self, nq: int, nv: int, nu: int, np: int = 0, input_bounds=None, q_init=None):
        """Initialize base model.
        
        Args:
            nq: Number of position coordinates
            nv: Number of velocity coordinates
            nu: Number of control inputs
            np: Number of parameters (default 0)
            input_bounds: Tuple of (min_array, max_array) or None for unbounded
            q_init: Initial position configuration (shape: nq,)
        """
        self.nq = nq
        self.nv = nv
        self.nx = nq + nv
        self.nu = nu
        self.np = np
        
        if input_bounds is None:
            self.input_min = -jnp.inf * jnp.ones(nu, dtype=jnp.float32)
            self.input_max = jnp.inf * jnp.ones(nu, dtype=jnp.float32)
        else:
            self.input_min = input_bounds[0]
            self.input_max = input_bounds[1]
        
        # Initial position - defaults to zeros if not provided
        if q_init is None:
            self._q_init = jnp.zeros(nq, dtype=jnp.float32)
        else:
            self._q_init = jnp.array(q_init, dtype=jnp.float32)
    
    @property
    def q_init(self) -> jnp.ndarray:
        """Initial position configuration."""
        return self._q_init
    
    @q_init.setter
    def q_init(self, value: jnp.ndarray):
        self._q_init = jnp.array(value, dtype=jnp.float32)
    
    @property
    def initial_state(self) -> jnp.ndarray:
        """Full initial state [q_init, zeros(nv)]."""
        return jnp.concatenate([self._q_init, jnp.zeros(self.nv, dtype=jnp.float32)])
    
    @property
    def input_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Input bounds as (min, max) tuple."""
        return (self.input_min, self.input_max)

    def integrate(self, state, inputs, dt):
        """Single step integration."""
        pass

    def integrate_sim(self, state, inputs, dt):
        """Integration for simulation."""
        return self.integrate(state, inputs, dt)

    def integrate_rollout(self, state, inputs, dt):
        """Batch integration for rollouts."""
        pass



class ModelParametric(BaseModel):
    def __init__(self, config: ModelConfig):
        """Initialize parametric model from configuration.
        
        Args:
            config: ModelConfig with dynamics function and integrator_type
        """
        # Extract all parameters from config and pass to BaseModel
        input_bounds = [config.input_min, config.input_max] if config.input_min is not None else None
        super().__init__(config.nq, config.nv, config.nu, config.np, input_bounds, config.q_init)

        self.dynamics_parametric = config.dynamics_fn
        integrator_type = config.integrator_type
        
        # Select integrator
        integrators = {
            MODEL_PARAMETRIC_INTEGRATOR_TYPES[0]: self.integrate_si_euler,
            MODEL_PARAMETRIC_INTEGRATOR_TYPES[1]: self.integrate_euler,
            MODEL_PARAMETRIC_INTEGRATOR_TYPES[2]: self.integrate_rk4,
            MODEL_PARAMETRIC_INTEGRATOR_TYPES[3]: config.dynamics_fn,
        }
        
        if integrator_type not in integrators:
            raise ValueError(f"Integrator '{integrator_type}' not supported. Available: {', '.join(integrators.keys())}")
        
        self.integrate_parametric = integrators[integrator_type]
        self.partial_sens_all = jax.jacfwd(self.integrate_parametric, argnums=(0, 1, 2))
        
        # Setup batch integration: vmap over states, inputs, and dt
        def _rollout_step(state_batch, input_batch, dt_batch, params_batch):
            return self.integrate_parametric(state_batch, input_batch, params_batch, dt_batch)
        
        self._integrate_rollout_fn = jax.jit(jax.vmap(_rollout_step, in_axes=(0, 0, None, None)))

    def integrate_rk4(self, state, inputs, params, dt: float):
        """
        One-step integration of the dynamics using Rk4 method
        """
        k1 = self.dynamics_parametric(state, inputs, params)
        k2 = self.dynamics_parametric(state + k1*dt/2., inputs, params)
        k3 = self.dynamics_parametric(state + k2 * dt / 2., inputs, params)
        k4 = self.dynamics_parametric(state + k3 * dt, inputs, params)
        return state + (dt/6.) * (k1 + 2. * k2 + 2. * k3 + k4)

    def integrate_euler(self, state, inputs, params, dt: float):
        """
        One-step integration of the dynamics using Euler method
        """
        return state + dt * self.dynamics_parametric(state, inputs, params)

    def integrate_si_euler(self, state, inputs, params, dt: float):
        """
        Semi-implicit Euler integration.
        As of now this is probably implemented inefficiently because the whole dynamics is evaluated two times.
        """
        v_kp1 = state[self.nq:] + dt * self.dynamics_parametric(state, inputs, params)[self.nq:]
        return jnp.concatenate([
                    state[:self.nq] + dt * self.dynamics_parametric(jnp.concatenate([state[:self.nq], v_kp1]), inputs, params)[:self.nq],
                    v_kp1])

    def sensitivity_step(self, state, inputs, params, state_sensitivity, input_sensitivity, dt):

        p_sens_all = self.partial_sens_all(state, inputs, params, dt)
        p_sens_state = p_sens_all[0]
        p_sens_inputs = p_sens_all[1]
        p_sens_params = p_sens_all[2]

        return p_sens_state @ state_sensitivity + p_sens_inputs @ input_sensitivity + p_sens_params
    
    def integrate_rollout(self, states, inputs, dt):
        """Batch integration: applies integrate_parametric to batched states/inputs."""
        params = jnp.array([])  # No parameters
        return self._integrate_rollout_fn(states, inputs, dt, params)


class Model(ModelParametric):
    def __init__(self, config: ModelConfig):
        """Create custom dynamics model from configuration.
        
        Args:
            config: ModelConfig with CUSTOM dynamics and integrator_type
        """
        if config.dynamics_model != DynamicsModel.CUSTOM:
            raise ValueError(f"Expected CUSTOM dynamics, got {config.dynamics_model}")
        
        super().__init__(config)
        self.nominal_parameters = jnp.array([])
        self.integrate_rollout_single = self.integrate
        
        # Load MuJoCo model for visualization if scene_path is provided
        self.mj_model = None
        self.mj_data = None
        if config.scene_path is not None:
            self.mj_model = mujoco.MjModel.from_xml_path(filename=config.scene_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            # Set initial qpos in visualization data
            self.mj_data.qpos = np.array(self._q_init)

    def integrate(self, state, inputs, dt):
        return self.integrate_parametric(state, inputs, self.nominal_parameters, dt)



class ModelMjx(BaseModel):
    def __init__(self, config: ModelConfig):
        """Create MJX dynamics model from configuration.
        
        Args:
            config: ModelConfig with MJX dynamics and scene_path
        """
        if config.dynamics_model != DynamicsModel.MJX:
            raise ValueError(f"Expected MJX dynamics, got {config.dynamics_model}")
        
        # Load MuJoCo model - this is where we discover actual dimensions
        self.mj_model = mujoco.MjModel.from_xml_path(filename=config.scene_path)
        self.kinematic = config.mjx_kinematic
        
        # Setup input bounds from config, or None for unbounded
        if config.input_min is None:
            input_bounds = None
        else:
            input_bounds = [config.input_min, config.input_max]
        
        # Get initial position: use config if provided, otherwise use MuJoCo model's default
        # MuJoCo stores default qpos in mj_model.qpos0
        if config.q_init is not None:
            q_init = config.q_init
        else:
            # Get default qpos from MuJoCo model and fix quaternion if needed
            q_init = jnp.array(self.mj_model.qpos0, dtype=jnp.float32)
            # If quaternion is all zeros (invalid), set to identity [0,0,0,1]
            if self.mj_model.nq >= 7 and jnp.allclose(q_init[3:7], 0.0):
                q_init = q_init.at[6].set(1.0)  # Set w component to 1
        
        # Extract actual dimensions from MuJoCo model (not from config!)
        # This ensures the model has correct dimensions regardless of config values
        super().__init__(
            nq=self.mj_model.nq,
            nv=self.mj_model.nv,
            nu=self.mj_model.nu,
            np=0,
            input_bounds=input_bounds,
            q_init=q_init
        )
        
        # Initialize MJX data structures
        self.mj_data = mujoco.MjData(self.mj_model)
        self.model = mjx.put_model(self.mj_model)
        self.data = mjx.put_data(self.mj_model, self.mj_data)
        
        # Set initial qpos in data structures
        self.set_qpos(self.q_init)
        
        # Setup integration methods based on kinematic flag
        if self.kinematic:
            self.integrate = jax.jit(self._integrate_kinematic)
            self.integrate_sim = jax.jit(self._integrate_kinematic_mjx)
            self.integrate_rollout = jax.jit(jax.vmap(self._integrate_kinematic, in_axes=(0, 0, None)))
            self.integrate_rollout_single = self._integrate_kinematic
        else:
            self.integrate = jax.jit(self._integrate_mjx)
            self.integrate_sim = self._integrate_mujoco
            self.integrate_rollout = jax.jit(jax.vmap(self._integrate, in_axes=(0, 0, None)))
            self.integrate_rollout_single = self._integrate

    @property
    def initial_state(self) -> jnp.ndarray:
        return self.data


    # here we need to work on data that is already on the gpu
    def _integrate_mjx(self, state: mjx.Data, inputs: jnp.array, dt: float):
        model = self.model.replace(opt=self.model.opt.replace(timestep=dt))
        state_next = state.replace(ctrl=inputs)
        state_next = mjx.step(model, state_next)
        return state_next

    def _integrate_mujoco(self, state: mujoco.MjData, inputs: jnp.array, dt: float):
        self.mj_model.opt.timestep = dt
        # Handle both MjData objects and JAX arrays
        if isinstance(state, mujoco.MjData):
            self.mj_data.qpos = state.qpos
            self.mj_data.qvel = state.qvel
        else:
            # state is a JAX array of shape (nq + nv,)
            self.mj_data.qpos = np.array(state[:self.mj_model.nq])
            self.mj_data.qvel = np.array(state[self.mj_model.nq:])
        self.mj_data.ctrl = inputs
        mujoco.mj_step(self.mj_model, self.mj_data)
        return jnp.concatenate([
            jnp.array(self.mj_data.qpos),
            jnp.array(self.mj_data.qvel)
        ])

    def _integrate(self, state: jnp.ndarray, inputs: jnp.array, dt: float):
        data_next = self.data
        data_next = data_next.replace(qpos=state[:self.model.nq], qvel=state[self.model.nq:], ctrl=inputs)
        data_next = mjx.step(self.model, data_next)
        return jnp.concatenate([data_next.qpos, data_next.qvel])

    def _integrate_kinematic(self, state: jnp.ndarray, inputs: jnp.array, dt: float):
        return state + dt * inputs

    def _integrate_kinematic_mjx(self, state: mjx.Data, inputs: jnp.array, dt: float):
        qpos = state.qpos
        qpos_next = qpos + dt * inputs
        state_next = self.data
        state_next = state_next.replace(qpos=qpos_next)
        return state_next

    def set_qpos(self, qpos):
        self.data = self.data.replace(qpos=qpos)
        self.mj_data.qpos = qpos



def create_model(
    config: ModelConfig,
) -> tuple[BaseModel, jnp.ndarray]:
    """Create a dynamics model from configuration.
    
    Args:
        config: ModelConfig containing model type and parameters
        
    Returns:
        Tuple of (model, initial_state_vector)
    """
    model_type = config.dynamics_model
    
    if model_type == DynamicsModel.CUSTOM:
        model = Model(config)
    elif model_type == DynamicsModel.MJX:
        model = ModelMjx(config)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    return model, model.initial_state