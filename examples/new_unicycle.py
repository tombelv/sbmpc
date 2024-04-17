import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sbmpc.new_model import Model, ModelJax
from sbmpc.solvers import SbMPC
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
import sbmpc.utils.simulation as simulation




input_max = jnp.array([2, 4])
input_min = -input_max

def unicycle_dynamics(self, state: jnp.array, inputs: jnp.array) -> jnp.array:
    state_dot = jnp.array([inputs[0]*jnp.cos(state[2]),
                            inputs[0]*jnp.sin(state[2]),
                            inputs[1]], dtype=jnp.float32)
    return state_dot


def cost_fn(state: jnp.array, state_ref: jnp.array, inputs: jnp.array) -> jnp.float32:
    """ Cost function to regulate the state to the desired value"""
    error = state - state_ref
    return 50*jnp.linalg.norm(error, ord=2) + jnp.linalg.norm(inputs, ord=2)


class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, num_iterations):
        super().__init__(initial_state, model, controller, num_iterations)

    def update(self):
        x_des = jnp.array([0, 0, 0], dtype=jnp.float32)
        # Compute the optimal input sequence
        time_start = time.time_ns()
        input_sequence = self.controller.compute_control_action(self.current_state, x_des).block_until_ready()
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[:self.model.nu]

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate_jit(self.current_state, ctrl, self.controller.dt)
        self.state_traj[self.iter + 1, :] = self.current_state


if __name__ == "__main__":

    system = Model(3, 3, 2, unicycle_dynamics) 
    jit_system = ModelJax(system, jax.devices('gpu')[0], "float32")

    x_init = jnp.array([2, 2, 0], dtype=jnp.float32)

    # dummy integration
    jit_system.integrate_jit(x_init, jnp.zeros(2), 0.0)

    mpc_config = ConfigMPC(0.02, 50, 0.1, num_parallel_computations=5000)
    gen_config = ConfigGeneral("float32", jax.devices("gpu")[0])

    solver = SbMPC(jit_system, cost_fn, mpc_config, gen_config)

    # Setup and run the simulation
    sim = Simulation(x_init, jit_system, solver, 600)
    sim.simulate()

    # Plot x-y position of the robot
    plt.plot(sim.state_traj[:, 0], sim.state_traj[:, 1])
    plt.show()
    # Plot the input trajectory
    plt.plot(sim.input_traj)
    plt.show()
