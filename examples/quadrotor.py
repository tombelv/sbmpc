import os

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sbmpc import BaseObjective
import sbmpc.settings as settings

from sbmpc.simulation import build_all
from sbmpc.geometry import skew, quat_product, quat2rotm, quat_inverse
from sbmpc.filter import MovingAverage

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
# Needed to remove warnings, to be investigated
jax.config.update("jax_default_matmul_precision", "high")

MODEL = "classic"
SCENE_PATH = "bitcraze_crazyflie_2/scene.xml"

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

    # def constraints(self, state, inputs, reference):
    #     return state[0] - 0.4

def get_quadroter_config() -> settings.Config:
    config = settings.Config()
    config.general.visualize = True
    config.MPC.dt = 0.02
    config.MPC.nu = 4
    config.MPC.horizon = 25
    config.MPC.std_dev_mppi = 0.2*jnp.array([0.1, 0.1, 0.1, 0.05])
    config.MPC.num_parallel_computations = 2000
    config.MPC.initial_guess = INPUT_HOVER
    config.MPC.lambda_mpc = 50.0
    config.MPC.smoothing = "Spline"
    config.MPC.num_control_points = 5
    config.MPC.gains = False

    config.robot.robot_scene_path = SCENE_PATH
    config.robot.nq = 7
    config.robot.nv = 6
    config.robot.nu = 4
    config.robot.input_min = INPUT_MIN
    config.robot.input_max = INPUT_MAX
    config.robot.q_init = jnp.array([0., 0., 0., 1., 0., 0., 0.], dtype=jnp.float32)  # hovering position

    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.MJX
    return config


if __name__ == "__main__":

    config = get_quadroter_config()

    # x_init = jnp.concatenate([config.robot[settings.ROBOT_Q_INIT_KEY],
    #                  jnp.zeros(config.robot[settings.ROBOT_NV_KEY], dtype=jnp.float32)], axis=0)
    # reference = jnp.concatenate((x_init, INPUT_HOVER))

    q_des = jnp.array([0.5, 0.5, 0.5, 1., 0., 0., 0.], dtype=jnp.float32)  # hovering position
    x_des = jnp.concatenate([q_des, jnp.zeros(config.robot.nv, dtype=jnp.float32)], axis=0)

    reference = jnp.concatenate((x_des, INPUT_HOVER))

    objective = Objective()

    sim = build_all(config, objective,
                    reference,
                    custom_dynamics_fn=quadrotor_dynamics)

    sim.simulate()

    time_vect = config.MPC.dt*jnp.arange(sim.state_traj.shape[0])
    ax = plt.figure().add_subplot(projection='3d')
    # Plot x-y-z position of the robot
    ax.plot(sim.state_traj[:, 0], sim.state_traj[:, 1], sim.state_traj[:, 2])
    plt.show()
    plt.plot(time_vect, sim.state_traj[:, 0:3])
    plt.legend(["x", "y", "z"])
    plt.grid()
    plt.show()
    # Plot the input trajectory
    plt.plot(time_vect[:-1], sim.input_traj)
    plt.legend(["F", "t_x", "t_y", "t_z"])
    plt.grid()

    plt.show()
