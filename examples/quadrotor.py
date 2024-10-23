import time, os

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sbmpc import Model, ModelMjx, SamplingBasedMPC, BaseObjective
from sbmpc.settings import Config
from sbmpc.simulation import Simulator
from sbmpc.geometry import skew, quat_product, quat2rotm, quat_inverse
from sbmpc.filter import MovingAverage

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
# Needed to remove warnings, to be investigated
jax.config.update("jax_default_matmul_precision", "high")

MODEL = "classic"

input_max = jnp.array([1, 2.5, 2.5, 2])
input_min = jnp.array([0, -2.5, -2.5, -2])

mass = 0.027
gravity = 9.81
inertia = jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5], dtype=jnp.float32)
inertia_mat = jnp.diag(inertia)

spatial_inertia_mat = jnp.diag(jnp.concatenate([mass*jnp.ones(3, dtype=jnp.float32), inertia]))
spatial_inertia_mat_inv = jnp.linalg.inv(spatial_inertia_mat)

input_hover = jnp.array([mass*gravity, 0., 0., 0.], dtype=jnp.float32)


@jax.jit
def quadrotor_dynamics(state, inputs, params):
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

    total_force = jnp.array([0., 0., inputs[0]]) - mass*gravity*orientation_mat[2, :]  # transpose + 3rd col = 3rd row

    total_torque = 1e-3*inputs[1:4] - skew(ang_vel) @ inertia_mat @ ang_vel  # multiplication by normalization factor

    acc = spatial_inertia_mat_inv @ jnp.concatenate([total_force, total_torque])

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
        ang_vel_err = state[10:] - state_ref[10:]

        return pos_err, att_err, vel_err, ang_vel_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:13]
        input_ref = reference[13:]
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        return (10 * pos_err.transpose() @ pos_err +
                0.5 * vel_err.transpose() @ vel_err +
                0.1 * ang_vel_err.transpose() @ ang_vel_err +
                (inputs-input_ref).transpose() @ jnp.diag(jnp.array([0.1, 0.1, 0.1, 0.5])) @ (inputs-input_ref))

    def final_cost(self, state, reference):
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        return (100 * pos_err.transpose() @ pos_err +
                1 * att_err.transpose() @ att_err +
                5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err)

    def constraints(self, state, inputs, reference):
        return state[0] - 0.4


class Simulation(Simulator):
    def __init__(self, initial_state, model, controller, num_iterations, visualization):
        super().__init__(initial_state, model, controller, num_iterations, visualization)

    def update(self):
        q_des = jnp.array([0.5, 0.5, 0.5, 1., 0., 0., 0.], dtype=jnp.float32)  # hovering position
        x_des = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0)

        reference = jnp.concatenate((x_des, input_hover))
        # Compute the optimal input sequence
        time_start = time.time_ns()
        input_sequence = self.controller.command(self.current_state_vec(), reference, num_steps=1).block_until_ready()
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[0, :]

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        self.state_traj[self.iter + 1, :] = self.current_state_vec()


if __name__ == "__main__":

    config = Config()
    config.MPC["dt"] = 0.02
    config.MPC["horizon"] = 25
    config.MPC["std_dev_mppi"] = 0.1*jnp.array([0.2, 0.3, 0.3, 0.15])
    config.MPC["num_parallel_computations"] = 500
    config.MPC["initial_guess"] = input_hover

    config.MPC["lambda"] = 2.0

    config.MPC["smoothing"] = "Spline"
    config.MPC["num_control_points"] = 5

    # config.MPC["gains"] = True

    if MODEL == "classic":
        system = Model(quadrotor_dynamics, nq=7, nv=6, nu=4, input_bounds=[input_min, input_max])
        q_init = jnp.array([0., 0., 0., 1., 0., 0., 0.], dtype=jnp.float32)  # hovering position
        x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
        state_init = x_init
    elif MODEL == "mjx":
        system = ModelMjx("bitcraze_crazyflie_2/scene.xml")
        q_init = system.data.qpos
        x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
        state_init = system.data
    else:
        raise ValueError("Model must be either 'classic' or 'mjx'")

    solver = SamplingBasedMPC(system, Objective(), config)

    reference = jnp.concatenate((x_init, input_hover))

    # dummy for jitting
    input_sequence = solver.command(x_init, reference, False).block_until_ready()

    # Setup and run the simulation
    sim = Simulation(state_init, system, solver, 500, False)
    sim.simulate()

    ax = plt.figure().add_subplot(projection='3d')
    # Plot x-y-z position of the robot
    ax.plot(sim.state_traj[:, 0], sim.state_traj[:, 1], sim.state_traj[:, 2])
    plt.show()
    plt.plot(sim.state_traj[:, 0:3])
    plt.legend(["x", "y", "z"])
    plt.grid()
    plt.show()
    # Plot the input trajectory
    plt.plot(sim.input_traj)
    plt.legend(["F", "t_x", "t_y", "t_z"])
    plt.show()
