import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sbmpc import Model, SamplingBasedMPC, BaseObjective
from sbmpc.settings import Config, DynamicsModel
from sbmpc.simulation import Simulator, build_all
from sbmpc.filter import MovingAverage


input_max = jnp.array([1, 1])
input_min = -input_max


@jax.jit
def unicycle_dynamics(state, inputs, params):
    state_dot = jnp.array([inputs[0] * jnp.cos(state[2]),
                           inputs[0] * jnp.sin(state[2]),
                           inputs[1]], dtype=jnp.float32)
    return state_dot


class Objective(BaseObjective):
    def running_cost(self, state: jnp.array, inputs: jnp.array, reference: jnp.array) -> jnp.float32:
        """ Cost function to regulate the state to the desired value"""
        error = state[:2] - reference[:2]
        return 1 * jnp.linalg.norm(error, ord=2) + jnp.linalg.norm(inputs, ord=2)

    def final_cost(self, state, reference):
        error = state - reference
        return 500 * jnp.linalg.norm(error, ord=2)


if __name__ == "__main__":

    config = Config()
    config.MPC.dt = 0.02
    config.MPC.horizon = 100
    config.MPC.nu = 2
    config.MPC.std_dev_mppi = jnp.array([0.1, 0.1])
    config.MPC.num_parallel_computations = 2000

    config.MPC.lambda_mpc = 5.0

    config.MPC.smoothing = "Spline"
    config.MPC.num_control_points = 5

    config.robot.nq = 3
    config.robot.nv = 0
    config.robot.nu = 2
    config.robot.input_min = input_min
    config.robot.input_max = input_max
    config.robot.q_init = jnp.array([2, 2, 0], dtype=jnp.float32)

    config.general.integrator_type = "rk4"

    config.solver_dynamics = DynamicsModel.CUSTOM
    config.sim_dynamics = DynamicsModel.CUSTOM

    objective = Objective()

    reference = jnp.array([0, 0, jnp.pi], dtype=jnp.float32)

    sim = build_all(config, objective,
                    reference,
                    custom_dynamics_fn=unicycle_dynamics)
    sim.simulate()

    # Plot x-y position of the robot
    plt.plot(sim.state_traj[:, 0], sim.state_traj[:, 1])
    plt.scatter(0, 0, marker='x')
    plt.show()
    # Plot the input trajectory
    plt.plot(sim.input_traj)
    plt.grid()
    plt.show()
