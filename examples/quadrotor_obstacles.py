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
from sbmpc.obstacle_loader import ObstacleLoader

jax.config.update("jax_default_matmul_precision", "high")

SCENE_PATH = "examples/bitcraze_crazyflie_2/obstacles.xml"

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
    """
    Simple quadrotor dynamics model with CoM placed at the geometric center

    Parameters
    ----------
    state : jnp.array
        state vector [pos (world frame),
                      attitude (unit quaternion [w, x, y, z]),
                      vel (world frame),
                      angular_velocity (body frame)]
    inputs : jnp.array):
        input vector [thrust (along the body-frame z axis), torque (body frame)]
    Returns
    -------
    state_dot :jnp.array
        time derivative of state with given inputs
    """

    quat = state[3:7]
    ang_vel = state[10:13]

    orientation_mat = quat2rotm(quat)
    ang_vel_quat = jnp.array([0., state[10], state[11], state[12]])

    total_force = jnp.array([0., 0., inputs[0]]) - MASS*GRAVITY*orientation_mat[2, :]  # transpose + 3rd col = 3rd row

    total_torque = 1e-3*inputs[1:4] - skew(ang_vel) @ INERTIA_MAT @ ang_vel  # multiplication by normalization factor

    acc = SPATIAL_INTERTIA_MAT_INV @ jnp.concatenate([total_force, total_torque])

    state_dot = jnp.concatenate([state[7:10],
                                 0.5 * quat_product(quat, ang_vel_quat),
                                 orientation_mat @ acc[:3],
                                 acc[3:6]])

    return state_dot


class Objective(BaseObjective):
    """ Cost function for the Quadrotor regulation task"""

    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        pos_err = state[0:3] - state_ref[0:3]
        att_err = quat_product(quat_inverse(state[3:7]), state_ref[3:7])[1:4]
        vel_err = state[7:10] - state_ref[7:10]
        ang_vel_err = state[10:13] - state_ref[10:13]

        return pos_err, att_err, vel_err, ang_vel_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:13]
        state_ref = state_ref.at[7:10].set(-1*(state[0:3] - state_ref[0:3]))
        input_ref = reference[13:13+4]
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        return (5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err +
                (inputs-input_ref).transpose() @ jnp.diag(jnp.array([10, 10, 10, 100])) @ (inputs-input_ref))

    def final_cost(self, state, reference):
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        return (10 * pos_err.transpose() @ pos_err +
                1 * att_err.transpose() @ att_err +
                5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err) 

    def constraints(self, state, inputs, reference):
        r = obsl.radius + 0.1                                           
        pos = state[0:3]
        n_obs = len(reference[17:])//3
        obs_pos = jnp.reshape(reference[17:],(n_obs,3))
        dist_from_obs = jnp.array([jnp.sum(abs(pos - obs) - r) for obs in obs_pos])  # l1 dist 
        # dist_from_obs = [jnp.where(jnp.sum(dist) < 0, jnp.positive(dist), jnp.negative(dist)) for dist in dist_from_obs]  # penalise only if x,y and z are in range
        return dist_from_obs * -1

    # def constraints(self, state, inputs, reference):
    #     return jnp.array([state[0] - 0.3, state[1] - 0.4])


if __name__ == "__main__":
    obsl = ObstacleLoader()
    obsl.create_obstacles()
    # Note: obstacles.xml is used directly as scene_path, no need to load into scene.xml

    model_config = ModelConfig(
        dynamics_model=DynamicsModel.CUSTOM,
        dynamics_fn=quadrotor_dynamics,
        nq=7,
        nv=6,
        nu=4,
        input_min=INPUT_MIN,
        input_max=INPUT_MAX,
        q_init=jnp.array([0., 0., 0., 1., 0., 0., 0.], dtype=jnp.float32),
        integrator_type="rk4",
        scene_path=SCENE_PATH,
    )

    controller_config = ControllerConfig(
        dt=0.02,
        horizon=25,
        num_samples=2000,
        lambda_inv=50.0,
        std_dev=0.2*jnp.array([0.1, 0.1, 0.1, 0.05]),
        initial_guess=INPUT_HOVER,
        smoothing="Spline",
        num_control_points=5,
        use_gains=False,
        device=jax.devices()[0],
        dtype=jnp.float32
    )

    simulation_config = SimulationConfig(
        dt=0.02,
        num_iterations=200,
        visualize=True,
    )

    # x_init = jnp.concatenate([robot_config[settings.ROBOT_Q_INIT_KEY],
    #                  jnp.zeros(robot_config[settings.ROBOT_NV_KEY], dtype=jnp.float32)], axis=0)
    # reference = jnp.concatenate((x_init, INPUT_HOVER))

    q_des = jnp.array([0.0, 1.5, 0.5, 1., 0., 0., 0.], dtype=jnp.float32)
    x_des = jnp.concatenate([q_des, jnp.zeros(model_config.nv, dtype=jnp.float32)], axis=0)

    horizon = controller_config.horizon + 1
    traj = obsl.get_obstacle_trajectory(simulation_config.num_iterations, "circle")[:horizon]

    reference = jnp.concatenate((x_des, INPUT_HOVER))
    reference = jnp.tile(reference, (horizon, 1))
    reference = jnp.concatenate([reference, traj], axis=1)

    objective = Objective()

    model, _ = create_model(model_config)
    controller = create_controller(controller_config, model, objective)
    sim = create_simulation(model, controller, simulation_config, Reference(reference))

    sim.run()

    obsl.reset_xmls()
 
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