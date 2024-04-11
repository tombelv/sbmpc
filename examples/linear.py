import jax
import jax.numpy as jnp
from jax import random

import sbmpc.model
from sbmpc.solvers import SbcMPC
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
import sbmpc.utils.simulation as simulation


class LinearModel(sbmpc.model.ModelJax):
    """ Linear model of a simple integrator"""
    def __init__(self, nx: int, nu: int, jax_device: jax.devices):
        super().__init__(nx, nu, jax_device)

    def dynamics(self, state: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        state_dot = 10*inputs
        return state_dot


def cost_fn(state: jnp.array, state_ref: jnp.array, inputs: jnp.array) -> jnp.float32:
    error = state - state_ref
    return 10*jnp.linalg.norm(error, ord=2) + jnp.linalg.norm(inputs, ord=2)


class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, num_iterations):
        super().__init__(initial_state, model, controller, num_iterations)

    def update(self):
        x_des = jnp.array([0, 0], dtype=jnp.float32)
        (input_sequence, _, _) = self.controller.compute_control_mppi(self.current_state,
                                                                          x_des,
                                                                          self.controller.best_control_parameters,
                                                                          self.controller.master_key)

        input = input_sequence[:self.model.nu]
        print("input:", input)
        self.current_state = self.model.integrate(self.current_state, input, self.controller.dt)

    def post_update(self):
        print("state:", self.current_state)


if __name__ == "__main__":

    device = jax.devices("gpu")[0]

    system = LinearModel(2, 2, device)

    x = jnp.array([2, 2], dtype=jnp.float32)

    # u = jnp.array([[1, 0], [2, 0]], dtype=jnp.float32)

    # key = random.PRNGKey(42)
    # input_vec = random.randint(key, (system.nu * 10,), minval=-2, maxval=2).reshape(10, system.nu)
    #
    # for _ in range(10):
    #     x = system.integrate_vectorized(x, u, 0.05)
    #     print(x)

    mpc_config = ConfigMPC(0.01, 20, 1, num_parallel_computations=10)
    gen_config = ConfigGeneral("float32", device)

    solver = SbcMPC(system, cost_fn, mpc_config, gen_config)

    sim = Simulation(x, system, solver, 100)

    sim.simulate()


