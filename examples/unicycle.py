import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sbmpc import (
    BaseObjective,
    ModelConfig,
    ControllerConfig,
    SimulationConfig,
    DynamicsModel,
    Reference,
    create_model,
    create_controller,
    create_simulation,
)


input_max = jnp.array([1, 1])
input_min = -input_max


@jax.jit
def unicycle_dynamics(state, inputs, params):
    state_dot = jnp.array([
        inputs[0] * jnp.cos(state[2]),
        inputs[0] * jnp.sin(state[2]),
        inputs[1]
    ], dtype=jnp.float32)
    return state_dot


class Objective(BaseObjective):
    def running_cost(self, state: jnp.array, inputs: jnp.array, reference: jnp.array) -> jnp.float32:
        error = state[:2] - reference[:2]
        return 1 * jnp.linalg.norm(error, ord=2) + jnp.linalg.norm(inputs, ord=2)

    def final_cost(self, state, reference):
        error = state - reference
        return 500 * jnp.linalg.norm(error, ord=2)


if __name__ == "__main__":
    model_config = ModelConfig(
        dynamics_model=DynamicsModel.CUSTOM,
        dynamics_fn=unicycle_dynamics,
        nq=3,
        nv=0,
        nu=2,
        input_min=input_min,
        input_max=input_max,
        q_init=jnp.array([2, 2, 0], dtype=jnp.float32),
        integrator_type="rk4",
    )

    controller_config = ControllerConfig(
        dt=0.02,
        horizon=100,
        num_samples=2000,
        lambda_inv=5.0,
        std_dev=jnp.array([0.1, 0.1]),
        smoothing="Spline",
        num_control_points=5,
        use_gains=False,
        device=jax.devices()[0],
        dtype=jnp.float32
    )

    simulation_config = SimulationConfig(
        dt=0.02,
        num_iterations=300,
        visualize=False,
    )

    objective = Objective()
    reference = Reference(jnp.array([0, 0, jnp.pi], dtype=jnp.float32))

    model, _ = create_model(model_config)
    controller = create_controller(controller_config, model, objective)
    sim = create_simulation(model, controller, simulation_config, reference)
    sim.run()

    states, controls = sim.get_trajectory()

    plt.plot(states[:, 0], states[:, 1])
    plt.scatter(0, 0, marker='x')
    plt.show()

    plt.plot(controls)
    plt.grid()
    plt.show()
