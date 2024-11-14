import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sbmpc import BaseObjective
from sbmpc.settings import Config, DynamicsModel, RobotConfig
from sbmpc.simulation import build_all


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

    robot_config = RobotConfig()
    robot_config.nq = 3
    robot_config.nv = 0
    robot_config.nu = 2
    robot_config.input_min = input_min
    robot_config.input_max = input_max
    robot_config.q_init = jnp.array([2, 2, 0], dtype=jnp.float32)

    config = Config(robot_config)

    config.MPC.dt = 0.02
    config.MPC.horizon = 100

    config.MPC.std_dev_mppi = jnp.array([0.1, 0.1])
    config.MPC.num_parallel_computations = 2000

    config.MPC.lambda_mpc = 5.0

    config.MPC.smoothing = "Spline"
    config.MPC.num_control_points = 5

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
