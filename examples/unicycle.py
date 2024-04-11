import jax
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt

from sbmpc.model import Model
from sbmpc.solvers import SbMPC
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
import sbmpc.utils.simulation as simulation


class Unicycle(Model):
    """ Kinematic model of a unicycle robot controlled with driving and steering velocities"""
    def __init__(self, nx: int, nu: int):
        super().__init__(nx, nu)

    def dynamics(self, state: jnp.array, inputs: jnp.array) -> jnp.array:
        state_dot = jnp.array([inputs[0]*jnp.cos(state[2]),
                               inputs[0]*jnp.sin(state[2]),
                               inputs[1]], dtype=jnp.float32)
        return state_dot


def cost_fn(state: jnp.array, state_ref: jnp.array, inputs: jnp.array) -> jnp.float32:
    error = state - state_ref
    return 50*jnp.linalg.norm(error, ord=2) + jnp.linalg.norm(inputs, ord=2)


class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, num_iterations):
        super().__init__(initial_state, model, controller, num_iterations)

    def update(self):
        x_des = jnp.array([0, 0, 0], dtype=jnp.float32)
        input_sequence = self.controller.compute_control_action(self.current_state, x_des)

        input = input_sequence[:self.model.nu]
        print("input:", input)
        self.input_traj[self.iter, :] = input
        self.current_state = self.model.integrate(self.current_state, input, self.controller.dt)

    def post_update(self):
        self.state_traj[self.iter, :] = self.current_state
        print("state:", self.current_state)


if __name__ == "__main__":

    device = jax.devices("gpu")[0]

    system = Unicycle(3, 2)

    x = jnp.array([2, 2, 0], dtype=jnp.float32)
    u = jnp.array([1, 0], dtype=jnp.float32)


    # key = random.PRNGKey(42)
    # input_vec = random.randint(key, (system.nu * 10,), minval=-2, maxval=2).reshape(10, system.nu)
    #
    # for _ in range(10):
    #     x = system.integrate_vectorized(x, u, 0.05)
    #     print(x)

    mpc_config = ConfigMPC(0.01, 20, 2, num_parallel_computations=10000)
    gen_config = ConfigGeneral("float32", device)

    solver = SbMPC(system, cost_fn, mpc_config, gen_config)

    sim = Simulation(x, system, solver, 1000)

    sim.simulate()

    # Plot x-y position of the robot
    plt.plot(sim.state_traj[:, 0], sim.state_traj[:, 1])
    plt.show()


