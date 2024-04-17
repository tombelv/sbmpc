import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import mujoco
from mujoco import mjx

class BaseModel(ABC):

    @abstractmethod
    def integrate(self, state, inputs, dt: float):
        pass
    @abstractmethod
    def setInitialState(self, state):
        pass

class Model(BaseModel):
    def __init__(self, nq: int,nv: int, nu: int,input_max, model_dynamics=None):
        self.nq=nq                        # number of generalized coordinates = dim(qpos)
        self.nv=nv                        # number of degrees of freedom = dim(qvel)
        self.nu=nu                        # number of control inputs
        self.input_max =input_max
        self.input_min = -self.input_max
        self.state0 = jnp.zeros(nq+nv, dtype=jnp.float32) # initstate
        self.model_dynamics = model_dynamics

    def setInitialState(self, state):
        pass

    def dynamics(self, state, inputs):
        x_dot = self.model_dynamics(state, inputs)
        return x_dot
    
    def integrate(self, state, inputs, dt: float):
        """ One-step integration of the dynamics using Rk4 method"""
        k1 = self.dynamics(state, inputs)
        k2 = self.dynamics(state + k1*dt/2., inputs)
        k3 = self.dynamics(state + k2 * dt / 2., inputs)
        k4 = self.dynamics(state + k3 * dt, inputs)
        return state + (dt/6.) * (k1 + 2. * k2 + 2. * k3 + k4)

class ModelMjx(BaseModel):
    def __init__(self, model_path, input_max):
        self.model_path = model_path
        # Load the MuJoCo model
        mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.nq=mj_model.nq                        # number of generalized coordinates = dim(qpos)
        self.nv=mj_model.nv                        # number of degrees of freedom = dim(qvel)
        self.nu=mj_model.nu                        # number of control inputs
        self.input_max =input_max
        self.input_min = -self.input_max
        mj_data = mujoco.MjData(mj_model)
        self.renderer = mujoco.Renderer(mj_model)
        self.mjx_model = mjx.put_model(mj_model)
        self.state0 = mjx.put_data(mj_model, mj_data) # initial state on the gpu

    def setInitialState(self, state):
        pass

    # here we need to work on data that are already on the gpu
    def integrate(self, state: mjx.MjData , inputs: jnp.array, dt: float):
        # here im applyng the control we need to check
        # TODO check how to change the control interface using mjx
        state.ctrl = inputs 
        mjx_data = mjx.step(self.mjx_model, state)
        return mjx_data

class ModelJax:
    def __init__(self,  other_model: BaseModel, device: jax.Device = jax.devices('cpu')[0], dtype_general="float32"):
        self.model = other_model
        self.device = device
        self.dtype_general = dtype_general
        self.input_max = self.model.input_max
        self.input_min = self.model.input_min
        # vectorized_integrate_jax = jax.vmap(self.integrate, in_axes=(0, 0, None), out_axes=0)
        # self.integrate_vectorized = jax.jit(vectorized_integrate_jax, device=self.device)

        self.integrate_jit = jax.jit(self.model.integrate, device=jax.devices('cpu')[0])

        # Make one dummy integration to compile during construction
        zero_ctrl = jnp.zeros(self.model.nu, dtype=dtype_general)
        self.integrate_jit(self.model.state0, zero_ctrl, 0.0)

        
