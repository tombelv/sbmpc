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
from sbmpc.geometry import skew, quat_product, quat2rotm, quat_inverse

jax.config.update("jax_default_matmul_precision", "high")

SCENE_PATH = "examples/bitcraze_crazyflie_2/scene.xml"

INPUT_MAX = jnp.array([1, 2.5, 2.5, 2])
INPUT_MIN = jnp.array([0, -2.5, -2.5, -2])

MASS = 0.027
GRAVITY = 9.81
INERTIA = jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5], dtype=jnp.float32)
INERTIA_MAT = jnp.diag(INERTIA)

SPATIAL_INTERTIA_MAT = jnp.diag(jnp.concatenate([MASS*jnp.ones(3, dtype=jnp.float32), INERTIA]))
SPATIAL_INTERTIA_MAT_INV = jnp.linalg.inv(SPATIAL_INTERTIA_MAT)

INPUT_HOVER = jnp.array([MASS*GRAVITY, 0., 0., 0.], dtype=jnp.float32)


@jax.jit
def quadrotor_dynamics(state: jnp.array, inputs: jnp.array, params: jnp.array) -> jnp.array:
    """Simple quadrotor dynamics model with CoM placed at the geometric center."""
    quat = state[3:7]
    ang_vel = state[10:13]

    orientation_mat = quat2rotm(quat)
    ang_vel_quat = jnp.array([0., state[10], state[11], state[12]])

    total_force = jnp.array([0., 0., inputs[0]]) - MASS*GRAVITY*orientation_mat[2, :]
    total_torque = 1e-3*inputs[1:4] - skew(ang_vel) @ INERTIA_MAT @ ang_vel

    acc = SPATIAL_INTERTIA_MAT_INV @ jnp.concatenate([total_force, total_torque])

    state_dot = jnp.concatenate([
        state[7:10],
        0.5 * quat_product(quat, ang_vel_quat),
        orientation_mat @ acc[:3],
        acc[3:6]
    ])

    return state_dot


class Objective(BaseObjective):
    """Cost function for the quadrotor regulation task."""

    def compute_state_error(self, state: jnp.ndarray, state_ref: jnp.ndarray):
        pos_err = state[0:3] - state_ref[0:3]
        att_err = quat_product(quat_inverse(state[3:7]), state_ref[3:7])[1:4]
        vel_err = state[7:10] - state_ref[7:10]
        ang_vel_err = state[10:13] - state_ref[10:13]

        return pos_err, att_err, vel_err, ang_vel_err

    def running_cost(self, state: jnp.ndarray, inputs: jnp.ndarray, reference):
        state_ref = reference[:13]
        state_ref = state_ref.at[7:10].set(-1*(state[0:3] - state_ref[0:3]))
        input_ref = reference[13:13+4]
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        return (
            5 * vel_err.transpose() @ vel_err +
            1 * ang_vel_err.transpose() @ ang_vel_err +
            (inputs-input_ref).transpose() @ jnp.diag(jnp.array([10, 10, 10, 100])) @ (inputs-input_ref)
        )

    def final_cost(self, state, reference):
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        return (
            10 * pos_err.transpose() @ pos_err +
            1 * att_err.transpose() @ att_err +
            5 * vel_err.transpose() @ vel_err +
            1 * ang_vel_err.transpose() @ ang_vel_err
        )


if __name__ == "__main__":
    # 1. Model configuration (custom dynamics, MuJoCo scene for visualization)
    model_config = ModelConfig(
        dynamics_model=DynamicsModel.CUSTOM,
        dynamics_fn=quadrotor_dynamics,
        nq=7,
        nv=6,
        nu=4,
        input_min=INPUT_MIN,
        input_max=INPUT_MAX,
        q_init=jnp.array([0., 0., 0.5, 1., 0., 0., 0.], dtype=jnp.float32),
        integrator_type="si_euler",
        scene_path=SCENE_PATH,
    )

    # 2. Controller configuration
    controller_config = ControllerConfig(
        dt=0.02,
        horizon=25,
        num_samples=1000,
        lambda_inv=50.0,
        std_dev=0.2*jnp.array([0.1, 0.1, 0.1, 0.05]),
        initial_guess=INPUT_HOVER,
        smoothing="Spline",
        num_control_points=5,
        use_gains=False,
        device=jax.devices()[0],
        dtype=jnp.float32
    )

    # 3. Simulation configuration
    simulation_config = SimulationConfig(
        dt=0.02,
        num_iterations=250,
        visualize=False,  # Uses scene_path from ModelConfig for visualization
    )

    # 4. Define objective and reference
    objective = Objective()

    q_des = jnp.array([0.5, 0.5, 0.5, 1., 0., 0., 0.], dtype=jnp.float32)
    x_des = jnp.concatenate([q_des, jnp.zeros(model_config.nv, dtype=jnp.float32)], axis=0)
    reference = Reference(jnp.concatenate((x_des, INPUT_HOVER)))

    # 5. Create model, controller, and simulation
    model, _ = create_model(model_config)
    controller = create_controller(controller_config, model, objective)
    sim = create_simulation(model, controller, simulation_config, reference)

    print("Running simulation...")
    sim.run()

    states, controls = sim.get_trajectory()
    time_vect = controller_config.dt * jnp.arange(states.shape[0])

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    plt.show()

    plt.plot(time_vect, states[:, 0:3])
    plt.legend(["x", "y", "z"])
    plt.grid()
    plt.show()

    plt.plot(time_vect[:-1], controls)
    plt.legend(["F", "t_x", "t_y", "t_z"])
    plt.grid()
    plt.show()
