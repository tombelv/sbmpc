"""
In this script we compare the mppi gains obtained from out differentiation procedure with the ones of an optimal LQR
controller.
We use the same Linear dynamics and cost of the LQR.
"""

import control

import jax.numpy as jnp

from sbmpc import BaseObjective
import sbmpc.settings as settings
from sbmpc.simulation import build_model_and_solver


# Number of tests to run
N_tests = 50

# Whether to use linearized dynamics or not in the mppi
LINEAR = False
# Whether to saturate the inputs for the mppi
INPUT_SATURATION = True

# Integration time step
DT = 0.05

MPPI_HORIZON = 25

# linearized dynamics of inverted pendulum
A = jnp.array([[0, 1], [1, 0]])
B = jnp.array([[0], [1]])

Q = jnp.array([[1, 0], [0, 1]])
R = Q[0, 0]

Ad = jnp.eye(2, 2) + DT * A
Bd = DT * B


K, S, E = control.dlqr(Ad, Bd, Q, R)
# Note that the feedback fains from the LQR are supposed to be applied like u = - K x
print("LQR gains: ", -K)

# Compute intial guess
x_init = jnp.array([jnp.pi/8, 0.0])
x_des = jnp.array([0.0, 0.0])
x = x_init.copy()
optimal_inputs = jnp.zeros((MPPI_HORIZON, 1))
for i in range(MPPI_HORIZON):
    u = -K @ (x - x_des)
    optimal_inputs = optimal_inputs.at[i, 0].set(u[0])
    x = Ad @ x + Bd @ u

print("Optimal inputs: ", optimal_inputs.T)

if LINEAR:
    def dynamics(x, u, p):
        return A @ x + B @ u
else:
    def dynamics(x, u, p):
        return jnp.array([x[1], jnp.sin(x[0]) + u[0]])


class Objective(BaseObjective):
    def running_cost(self, state, inputs, reference):
        return ((state - reference).T @ Q @ (state - reference)) / DT

    def final_cost(self, state, reference):
        return (state - reference).T @ S @ (state - reference) / DT


if __name__ == "__main__":

    robot_config = settings.RobotConfig()

    robot_config.nq = 1
    robot_config.nv = 1
    robot_config.nu = 1
    if INPUT_SATURATION:
        robot_config.input_min = jnp.array([-0.5], dtype=jnp.float32)
        robot_config.input_max = jnp.array([0.5], dtype=jnp.float32)

    robot_config.q_init = jnp.array([0.], dtype=jnp.float32)  # hovering position

    config = settings.Config(robot_config)

    config.integrator_type = "euler"

    config.MPC.dt = DT
    config.MPC.horizon = MPPI_HORIZON
    config.MPC.std_dev_mppi = jnp.array([0.5])
    config.MPC.num_parallel_computations = 1000
    config.MPC.lambda_mpc = 2.0
    config.MPC.num_control_points = config.MPC.horizon
    config.MPC.gains = True

    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM

    objective = Objective()

    model, solver = build_model_and_solver(config, objective, custom_dynamics_fn=dynamics)

    print("Optimal LQR cost: ", objective.final_cost(x_init, x_des))

    # perform a first dummy iteration to initialize the solver
    solver.sampler.optimal_samples = optimal_inputs
    input = solver.command(x_init, x_des, False, num_steps=1).block_until_ready()


    error_norms = []
    mppi_gains = []

    for i in range(N_tests):
        solver.sampler.optimal_samples = optimal_inputs
        input = solver.command(x_init, x_des, False, num_steps=1).block_until_ready()

        print(input.T)

        mppi_gains.append(solver.gains[0])
        print(f"Test {i+1}/{N_tests}, MPPI gains: {mppi_gains[-1]}")

        error_norms.append(jnp.linalg.norm(mppi_gains + K, jnp.inf))

    # jnp.save(f"mppi_gains_sat_hard{config.MPC.num_parallel_computations}", jnp.array(mppi_gains))

    print("mean error norm: ", jnp.mean(jnp.array(error_norms)))
    print("max error norm: ", jnp.max(jnp.array(error_norms)))
