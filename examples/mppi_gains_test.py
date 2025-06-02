"""
In this script we compare the mppi gains obtained from out differentiation procedure with the ones of an optimal LQR
controller.
We use the same Linear dynamics and cost of the LQR.
"""

import control
import time, os

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sbmpc import Model, BaseObjective
import sbmpc.settings as settings
from sbmpc.simulation import build_model_and_solver


# Simple double integrator model
A = jnp.array([[0, 1], [0, 0]])
B = jnp.array([[0], [1]])

Q = jnp.array([[1, 0], [0, 1]])
R = Q[0, 0]

Ad = jnp.eye(2, 2) + 0.05 * A
Bd = 0.05*B


K, S, E = control.dlqr(Ad, Bd, Q, R)
# Note that the feedback fains from the LQR are supposed to be applied like u = -K x
print("LQR gains: ", -K)

x = jnp.array([0.0, 0.0])
x_des = jnp.array([0.5, 0.0])
optimal_inputs = jnp.zeros((25, 2))
for i in range(25):
    u = -K @ (x - x_des)
    optimal_inputs = optimal_inputs.at[i, 0].set(u[0])
    x = Ad @ x + Bd @ u


# Redefine B matrix since mppi does not support single input systems (to be fixed)
B_mppi = jnp.array([[0, 0], [1, 0]])

def dynamics(x, u, p):
    return A @ x + B_mppi @ u


class Objective(BaseObjective):
    def running_cost(self, state, inputs, reference):
        return 20*((state - reference).T @ Q @ (state - reference))

    def final_cost(self, state, reference):
        return 20*(state - reference).T @ S @ (state - reference)


if __name__ == "__main__":

    robot_config = settings.RobotConfig()

    robot_config.nq = 1
    robot_config.nv = 1
    robot_config.nu = 2

    robot_config.q_init = jnp.array([0.], dtype=jnp.float32)  # hovering position

    config = settings.Config(robot_config)

    config.integrator_type = "euler"

    config.MPC.dt = 0.05
    config.MPC.horizon = 25
    config.MPC.std_dev_mppi = jnp.array([0.5, 0.0])
    config.MPC.num_parallel_computations = 10000
    config.MPC.lambda_mpc = 2.0
    config.MPC.num_control_points = config.MPC.horizon
    config.MPC.gains = True

    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM

    objective = Objective()

    model, solver = build_model_and_solver(config, objective, custom_dynamics_fn=dynamics)

    solver.sampler.optimal_samples = optimal_inputs

    input = solver.command(jnp.array([0.0, 0.0]), jnp.array([0.5, 0.0]), False, num_steps=1).block_until_ready()

    print(input[0])
    print("optimal inputs: ", optimal_inputs[0])

    mppi_gains = solver.gains[0]

    print("MPPI gains: ", mppi_gains)


    print("error norm: ", jnp.linalg.norm(mppi_gains + K, jnp.inf))


