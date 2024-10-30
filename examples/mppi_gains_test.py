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

from sbmpc import Model, ModelMjx, SamplingBasedMPC, BaseObjective
from sbmpc.settings import Config


jax.config.update("jax_default_matmul_precision", "high")


# Simple double integrator model
A = jnp.array([[0, 1], [0, 0]])
B = jnp.array([[0, 0], [1, 1]])

Q = jnp.array([[1, 0], [0, 1]])
R = Q

Ad = jnp.eye(2, 2) + 0.05 * A
Bd = 0.05*B

# K, S, E = control.lqr(A, B, Q, R)
K, S, E = control.dlqr(Ad, Bd, Q, R)
# Note that the feedback fains from the LQR are supposed to be applied like u = -K x
print(-K)

def dynamics(x, u, p):
    return A @ x + B @ u


class Objective(BaseObjective):
    def running_cost(self, state, inputs, reference):
        return 20*(state - reference).T @ Q @ (state - reference) + inputs.T @ R @ inputs

    def final_cost(self, state, reference):
        return 20*(state - reference).T @ S @ (state - reference)


if __name__ == "__main__":

    config = Config()
    config.MPC["dt"] = 0.05
    config.MPC["horizon"] = 25
    config.MPC["std_dev_mppi"] = jnp.array([0.5, 0.5])
    # config.MPC["initial_guess"] = - K @ jnp.array([0.5, 0.0])
    config.MPC["num_parallel_computations"] = 10000

    config.MPC["lambda"] = 1.0

    config.MPC["smoothing"] = "Spline"
    config.MPC["num_control_points"] = 5

    config.MPC["gains"] = True

    system = Model(dynamics, nq=1, nv=1, nu=2, integrator_type="euler")
    objective = Objective()

    solver = SamplingBasedMPC(system, objective, config)

    print("optimal final cost 1 ", 0.05*objective.final_cost(jnp.array([0.0, 0.0]), jnp.array([0.5, 0.0])))
    # dummy for jitting
    action1 = solver.command(jnp.array([0.0, 0.0]), jnp.array([0.5, 0.0]), False, num_steps=5).block_until_ready()
    print(action1[0])
    print(solver.gains)

