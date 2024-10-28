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
        if jnp.array_equal(input_bounds, (-jnp.inf, jnp.inf)):
            self.input_min = input_bounds[0]*jnp.ones(self.nu, dtype=jnp.float32)
            self.input_max = input_bounds[1]*jnp.ones(self.nu, dtype=jnp.float32)
        else:
            self.input_min = input_bounds[0]
            self.input_max = input_bounds[1]
    
    def get_nq(self):
        return self.nq

    def integrate(self, state, inputs, dt):
        pass

    def integrate_rollout(self, state, inputs, dt):
        pass



class Model(BaseModel):
    def __init__(self, model_dynamics,
                 nq: int,
                 nv: int,
                 nu: int,
                 input_bounds=(-jnp.inf, jnp.inf),
                 integrator_type="si_euler"):

        super().__init__(nq, nv, nu, input_bounds)

        self.state0 = jnp.zeros(self.nx, dtype=jnp.float32)  # initstate
        self.dynamics = model_dynamics

        if integrator_type == "si_euler":
            self.integrate = self.integrate_si_euler
        elif integrator_type == "euler":
            self.integrate = self.integrate_euler
        elif integrator_type == "rk4":
            self.integrate = self.integrate_rk4
        elif integrator_type == "custom_discrete":
            self.integrate = model_dynamics
        else:
            raise ValueError("""
            Integrator type not supported.
            Available types: si_euler, euler, rk4, custom_discrete
            """)

        integrate_vect = jax.vmap(self.integrate, in_axes=(0, 0, None))
        self.integrate_rollout = jax.jit(integrate_vect)



    def integrate_rk4(self, state, inputs, dt: float):
        """
        One-step integration of the dynamics using Rk4 method
        """
        k1 = self.dynamics(state, inputs)
        k2 = self.dynamics(state + k1*dt/2., inputs)
        k3 = self.dynamics(state + k2 * dt / 2., inputs)
        k4 = self.dynamics(state + k3 * dt, inputs)
        return state + (dt/6.) * (k1 + 2. * k2 + 2. * k3 + k4)

    def integrate_euler(self, state, inputs, dt: float):
        """
        One-step integration of the dynamics using Euler method
        """
        return state + dt * self.dynamics(state, inputs)

    def integrate_si_euler(self, state, inputs, dt: float):
        """
        Semi-implicit Euler integration.

        As of now this is probably implemented inefficiently because the whole dynamics is evaluated two times.
        """
        v_kp1 = state[self.nq:] + dt * self.dynamics(state, inputs)[self.nq:]
        return jnp.concatenate([
                    state[:self.nq] + dt * self.dynamics(jnp.concatenate([state[:self.nq], v_kp1]), inputs)[:self.nq],
                    v_kp1])


class ModelMjx(BaseModel):
    def __init__(self, model_path, kinematic=False, input_bounds=(-jnp.inf, jnp.inf)):
        self.model_path = model_path
        # Load the MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(filename=self.model_path)

        if kinematic and jnp.array_equal(input_bounds, (-jnp.inf, jnp.inf)):
            input_bounds = [-jnp.inf * jnp.ones(self.mj_model.nu, dtype=jnp.float32),
                             jnp.inf * jnp.ones(self.mj_model.nu, dtype=jnp.float32)]
        elif not kinematic:
            input_bounds = (self.mj_model.actuator_ctrlrange[:, 0], self.mj_model.actuator_ctrlrange[:, 1])
        else:
            input_bounds = [input_bounds[0], input_bounds[1]]

        super().__init__(self.mj_model.nq, self.mj_model.nv, self.mj_model.nu, input_bounds)
        self.mj_data = mujoco.MjData(self.mj_model)
        # self.renderer = mujoco.Renderer(mj_model)
        self.model = mjx.put_model(self.mj_model)
        self.data = mjx.put_data(self.mj_model, self.mj_data)  # initial state on the gpu

        # If the model is set as kinematic, it follows the dynamics $\dot q = u$
        if kinematic:
            integrate_vect = jax.vmap(self._integrate_kinematic, in_axes=(0, 0, None))
            self.integrate_rollout = jax.jit(integrate_vect)
            self.integrate = jax.jit(self._integrate_kinematic)
        else:
            integrate_vect = jax.vmap(self._integrate, in_axes=(0, 0, None))
            self.integrate_rollout = jax.jit(integrate_vect)
            self.integrate = jax.jit(self._integrate_mjx)


    # here we need to work on data that are already on the gpu
    def _integrate_mjx(self, state: mjx.Data, inputs: jnp.array, dt: float):
        state_next = state.replace(ctrl=inputs)
        state_next = mjx.step(self.model, state_next)
        return state_next

    def _integrate(self, state: jnp.ndarray, inputs: jnp.array, dt: float):
        data_next = self.data
        data_next = data_next.replace(qpos=state[:self.model.nq], qvel=state[self.model.nq:], ctrl=inputs)
        data_next = mjx.step(self.model, data_next)
        return jnp.concatenate([data_next.qpos, data_next.qvel])

    def _integrate_kinematic(self, state: jnp.ndarray, inputs: jnp.array, dt: float):
        return state + dt * inputs

    def set_qpos(self, qpos):
        self.data = self.data.replace(qpos=qpos)
        self.mj_data.qpos = qpos



        
