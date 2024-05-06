import time, os

import jax
import jax.numpy as jnp

import numpy

import matplotlib.pyplot as plt

import mujoco.mjx as mjx

from sbmpc.model import ModelMjx
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
import sbmpc.utils.simulation as simulation
from sbmpc.utils.geometry import quat_product, quat_inverse

# This should actually be taken from the model
input_hover = jnp.array([0.027*9.81, 0., 0., 0.], dtype=jnp.float32)

os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
    )

class Objective(BaseObjective):
    """ Cost function for the Quadrotor regulation task"""
    def compute_state_error(self, mjx_state: mjx.Data, state_ref: jnp.array) -> jnp.array:
        state = jnp.concatenate([mjx_state.qpos, mjx_state.qvel])

        pos_err = state[0:3] - state_ref[0:3]
        att_err = quat_product(quat_inverse(state[3:7]), state_ref[3:7])[1:4]
        vel_err = state[7:10] - state_ref[7:10]
        ang_vel_err = state[10:] - state_ref[10:]

        return pos_err, att_err, vel_err, ang_vel_err

    def running_cost(self, mjx_state: mjx.Data, state_ref: jnp.array, inputs: jnp.array) -> jnp.float32:
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(mjx_state, state_ref)
        return (10 * pos_err.transpose() @ pos_err +
                0.01 * att_err.transpose() @ att_err +
                0.5 * vel_err.transpose() @ vel_err +
                0.1 * ang_vel_err.transpose() @ ang_vel_err +
                (inputs-input_hover).transpose() @ jnp.diag(jnp.array([1, 0.1, 0.1, 0.5])) @ (inputs-input_hover) )

    def final_cost(self, mjx_state: mjx.Data, state_ref):
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(mjx_state, state_ref)
        return (100 * pos_err.transpose() @ pos_err +
                0.1 * att_err.transpose() @ att_err +
                5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err)

class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, num_iterations):
        super().__init__(initial_state, model, controller, num_iterations)

    def update(self):
        q_des = jnp.array([0.5, 0.5, 0.5, 1., 0., 0., 0.], dtype=jnp.float32)
        x_des = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0)
        # Compute the optimal input sequence
        time_start = time.time_ns()
        input_sequence = self.controller.compute_control_action(self.current_state, x_des).block_until_ready()
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[:self.model.nu]

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        self.state_traj[self.iter + 1, :] = numpy.concatenate([self.current_state.qpos, self.current_state.qvel])


if __name__ == "__main__":

    system = ModelMjx("bitcraze_crazyflie_2/cf2.xml")

    q_init = jnp.array([0.0, 0.0, 0., 1., 0., 0., 0.], dtype=jnp.float32)  # hovering position

    x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)

    mpc_config = ConfigMPC(system.model.opt.timestep,
                           25,
                           jnp.array([0.2, 0.2, 0.2, 0.2]),
                           num_parallel_computations=10000,
                           initial_guess=input_hover)
    gen_config = ConfigGeneral("float32", jax.devices("gpu")[0])

    solver = SbMPC(system, Objective(), mpc_config, gen_config)

    # dummy for jitting
    input_sequence = solver.compute_control_action(system.data, x_init).block_until_ready()

    # Setup and run the simulation
    sim = Simulation(system.data, system, solver, 500)
    sim.simulate()

    ax = plt.figure().add_subplot(projection='3d')
    # Plot x-y-z position of the robot
    ax.plot(sim.state_traj[:, 0], sim.state_traj[:, 1], sim.state_traj[:, 2])
    plt.show()
    plt.plot(sim.state_traj[:, 0:3])
    plt.legend(["x", "y", "z"])
    plt.show()
    # Plot the input trajectory
    plt.plot(sim.input_traj)
    plt.legend(["F", "t_x", "t_y", "t_z"])
    plt.show()


