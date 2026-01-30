"""
In this script we compare the MPPI gains obtained from our differentiation procedure
with the ones of an optimal LQR controller.
We use the same linear dynamics and cost of the LQR.
"""

import control
import jax
import jax.numpy as jnp

from sbmpc import (
    BaseObjective,
    ModelConfig,
    ControllerConfig,
    DynamicsModel,
    create_model,
    create_controller,
)


# Simple double integrator model
A = jnp.array([[0, 1], [0, 0]])
B = jnp.array([[0], [1]])

Q = jnp.array([[1, 0], [0, 1]])
R = Q[0, 0]

Ad = jnp.eye(2, 2) + 0.05 * A
Bd = 0.05 * B


K, S, E = control.dlqr(Ad, Bd, Q, R)
print("LQR gains: ", -K)

x = jnp.array([0.0, 0.0])
x_des = jnp.array([0.5, 0.0])
optimal_inputs = jnp.zeros((25, 2))
for i in range(25):
    u = -K @ (x - x_des)
    optimal_inputs = optimal_inputs.at[i, 0].set(u[0])
    x = Ad @ x + Bd @ u


# Redefine B matrix since MPPI does not support single input systems (to be fixed)
B_mppi = jnp.array([[0, 0], [1, 0]])

def dynamics(x, u, p):
    return A @ x + B_mppi @ u


class Objective(BaseObjective):
    def running_cost(self, state, inputs, reference):
        return 20 * ((state - reference).T @ Q @ (state - reference))

    def final_cost(self, state, reference):
        return 20 * (state - reference).T @ S @ (state - reference)


if __name__ == "__main__":
    model_config = ModelConfig(
        dynamics_model=DynamicsModel.CUSTOM,
        dynamics_fn=dynamics,
        nq=1,
        nv=1,
        nu=2,
        q_init=jnp.array([0.], dtype=jnp.float32),
        integrator_type="euler",
    )

    controller_config = ControllerConfig(
        dt=0.05,
        horizon=25,
        num_samples=10000,
        lambda_inv=2.0,
        std_dev=jnp.array([0.5, 0.0]),
        smoothing=None,
        num_control_points=25,
        use_gains=True,
        device=jax.devices()[0],
        dtype=jnp.float32
    )

    objective = Objective()

    model, _ = create_model(model_config)
    controller = create_controller(controller_config, model, objective)

    controller.sampler.optimal_samples = optimal_inputs

    _ = controller.compute_control(jnp.array([0.0, 0.0]), jnp.array([0.5, 0.0]), num_iterations=1)

    mppi_gains = controller.gains[0]

    print("MPPI gains: ", mppi_gains)


