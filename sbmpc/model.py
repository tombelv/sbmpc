import jax
import jax.numpy as jnp


class Model:
    def __init__(self, nx: int, nu: int):
        self.nx = nx    # State dimension
        self.nu = nu    # Input dimension
        self.input_max = 1e10*jnp.ones(nu, dtype=jnp.float32)
        self.input_min = -self.input_max

    def dynamics(self, state, inputs):
        """ Implements the continuous time dynamics equation"""
        raise NotImplementedError

    def integrate(self, state, inputs, dt: float):
        """ One-step integration of the dynamics using Rk4 method"""
        k1 = self.dynamics(state, inputs)
        k2 = self.dynamics(state + k1*dt/2., inputs)
        k3 = self.dynamics(state + k2 * dt / 2., inputs)
        k4 = self.dynamics(state + k3 * dt, inputs)
        return state + (dt/6.) * (k1 + 2. * k2 + 2. * k3 + k4)


class ModelJax(Model):
    def __init__(self, nx: int, nu: int, device: jax.Device = jax.devices('cpu')[0], dtype_general="float32"):
        super().__init__(nx, nu)
        self.device = device
        self.dtype_general = dtype_general

        # vectorized_integrate_jax = jax.vmap(self.integrate, in_axes=(0, 0, None), out_axes=0)
        # self.integrate_vectorized = jax.jit(vectorized_integrate_jax, device=self.device)

        self.integrate_jit = jax.jit(self.integrate, device=jax.devices('cpu')[0])

        # Make one dummy integration to compile during construction
        zero_state = jnp.zeros(nx, dtype=dtype_general)
        zero_ctrl = jnp.zeros(nu, dtype=dtype_general)
        self.integrate_jit(zero_state, zero_ctrl, 0.0)




