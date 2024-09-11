import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sbmpc.model import Model
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
import sbmpc.utils.simulation as simulation
import sbmpc.utils.filter as fltr


input_max = jnp.array([2, 4])
input_min = -input_max


@jax.jit
def unicycle_dynamics(state: jnp.array, inputs: jnp.array) -> jnp.array:
    state_dot = jnp.array([inputs[0] * jnp.cos(state[2]),
                           inputs[0] * jnp.sin(state[2]),
                           inputs[1]], dtype=jnp.float32)
    return state_dot


class Objective(BaseObjective):
    def running_cost(self, state: jnp.array, inputs: jnp.array, reference: jnp.array) -> jnp.float32:
        """ Cost function to regulate the state to the desired value"""
        error = state[:2] - reference[:2]
        return 5 * jnp.linalg.norm(error, ord=2) + jnp.linalg.norm(inputs, ord=2)

    def final_cost(self, state, reference):
        error = state - reference
        return 500 * jnp.linalg.norm(error, ord=2)


class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, num_iterations):
        super().__init__(initial_state, model, controller, num_iterations)

    def update(self):
        x_des = jnp.array([0, 0, jnp.pi], dtype=jnp.float32)
        # Compute the optimal input sequence
        time_start = time.time_ns()
        input_sequence = self.controller.command(self.current_state, x_des, shift_guess=True).block_until_ready()
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[:self.model.nu]

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        self.state_traj[self.iter + 1, :] = self.current_state


if __name__ == "__main__":

    system = Model(unicycle_dynamics, nq=3, nv=0, nu=2, input_bounds=[input_min, input_max], integrator_type="rk4")

    x_init = jnp.array([2, 2, 0], dtype=jnp.float32)

    mpc_config = ConfigMPC(dt=0.02, horizon=50, std_dev_mppi=jnp.array([0.8, 0.75]), num_parallel_computations=5000)
    window_size = 3
    mpc_config.filter = fltr.MovingAverage(window_size=window_size, step_size=system.nu)

    gen_config = ConfigGeneral("float32", jax.devices("gpu")[0])

    solver = SbMPC(system, Objective(), mpc_config, gen_config)

    # Setup and run the simulation
    sim = Simulation(x_init, system, solver, 500)
    sim.simulate()

    # Plot x-y position of the robot
    plt.plot(sim.state_traj[:, 0], sim.state_traj[:, 1])
    plt.scatter(0, 0, marker='x')
    plt.show()
    # Plot the input trajectory
    plt.plot(sim.input_traj)
    plt.show()
