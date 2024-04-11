import jax
import jax.numpy as jnp



class Model:
    def __init__(self, nx: int, nu: int):
        self.nx = nx    # State dimension
        self.nu = nu    # Input dimension

    def dynamics(self, state, inputs):
        """ Implements the continuous time dynamics equation"""
        raise NotImplementedError

    def integrate(self, state, inputs, dt: float):
        """ One-step integration of the dynamics using Euler method"""
        return state + dt * self.dynamics(state, inputs)


class ModelJax(Model):
    def __init__(self, nx: int, nu: int, device: jax.Device = jax.devices('cpu')[0], dtype_general="float32"):
        super().__init__(nx, nu)
        self.device = device
        self.dtype_general = dtype_general

        vectorized_integrate_jax = jax.vmap(self.integrate, in_axes=(0, 0, None), out_axes=0)
        self.integrate_vectorized = jax.jit(vectorized_integrate_jax, device=self.device)




