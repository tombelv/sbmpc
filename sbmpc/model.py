import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import mujoco
from mujoco import mjx


class BaseModel(ABC):
    def __init__(self, nq: int, nv: int, nu: int, input_bounds=(-jnp.inf, jnp.inf)):
        self.nq = nq  # number of generalized coordinates = dim(qpos)
        self.nv = nv  # number of degrees of freedom = dim(qvel)
        self.nx = nq + nv
        self.nu = nu  # number of control inputs
        if input_bounds is (-jnp.inf, jnp.inf):
            self.input_min = input_bounds[0]*jnp.ones(self.nu, dtype=jnp.float32)
            self.input_max = input_bounds[1]*jnp.ones(self.nu, dtype=jnp.float32)
        else:
            self.input_min = input_bounds[0]
            self.input_max = input_bounds[1]

    @abstractmethod
    def integrate(self, state, inputs, dt: float):
        pass

    @abstractmethod
    def setInitialState(self, state):
        pass


class Model(BaseModel):
    def __init__(self, model_dynamics, nq: int, nv: int, nu: int, input_bounds=(-jnp.inf, jnp.inf)):
        super().__init__(nq, nv, nu, input_bounds)

        self.state0 = jnp.zeros(self.nx, dtype=jnp.float32)  # initstate
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
    def __init__(self, model_path):
        self.model_path = model_path
        # Load the MuJoCo model
        mj_model = mujoco.MjModel.from_xml_path(self.model_path)

        input_bounds = (-jnp.inf*jnp.ones(mj_model.nu, dtype=jnp.float32),
                        jnp.inf*jnp.ones(mj_model.nu, dtype=jnp.float32))  # We should take input bounds from the model

        super().__init__(mj_model.nq, mj_model.nv, mj_model.nu, input_bounds)
        mj_data = mujoco.MjData(mj_model)
        # self.renderer = mujoco.Renderer(mj_model)
        self.mjx_model = mjx.put_model(mj_model)
        self.state0 = mjx.put_data(mj_model, mj_data)  # initial state on the gpu

    def setInitialState(self, state):
        pass

    # here we need to work on data that are already on the gpu
    def integrate(self, state: mujoco.MjData, inputs: jnp.array, dt: float):
        # here im applyng the control we need to check
        # TODO check how to change the control interface using mjx
        state.ctrl = inputs 
        mjx_data = mjx.step(self.mjx_model, state)
        return mjx_data


# Maybe this should just become a decorator  Model and ModelMjx
# class ModelJax:
#     def __init__(self,  other_model: Model, device: jax.Device = jax.devices('cpu')[0], dtype_general="float32"):
#         self.model = other_model
#         self.device = device
#         self.dtype_general = dtype_general
#         # self.input_max = self.model.input_max
#         # self.input_min = self.model.input_min
#
#         self.integrate = jax.jit(self.model.integrate, device=device)
#
#         # Make one dummy integration to compile during construction
#         zero_ctrl = jnp.zeros(self.model.nu, dtype=dtype_general)
#         self.integrate(self.model.state0, zero_ctrl, 0.0)

        
