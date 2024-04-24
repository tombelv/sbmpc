import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sbmpc.model import ModelMjx
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
import sbmpc.utils.simulation as simulation
from sbmpc.utils.geometry import quat_product, quat_inverse

# This should actually be taken from the model
input_hover = jnp.array([0.027*9.81, 0., 0., 0.], dtype=jnp.float32)


class Objective(BaseObjective):
    def running_cost(self, state: jnp.array, state_ref: jnp.array, inputs: jnp.array) -> jnp.float32:
        """ Cost function to regulate the state to the desired value"""
        pos_err = state[0:3] - state_ref[0:3]
        att_err = quat_product(quat_inverse(state[3:7]), state_ref[3:7])[1:4]
        vel_err = state[7:10] - state_ref[7:10]
        ang_vel_err = state[10:] - state_ref[10:]

        return (10 * pos_err.transpose() @ pos_err +
                0.01 * att_err.transpose() @ att_err +
                0.5 * vel_err.transpose() @ vel_err +
                0.01 * ang_vel_err.transpose() @ ang_vel_err +
                (inputs-input_hover).transpose() @ jnp.diag(jnp.array([1, 0.01, 0.01, 0.5])) @ (inputs-input_hover) )

    def final_cost(self, state, state_ref):
        pos_err = state[0:3] - state_ref[0:3]
        att_err = quat_product(quat_inverse(state[3:7]), state_ref[3:7])[1:4]
        vel_err = state[7:10] - state_ref[7:10]
        ang_vel_err = state[10:] - state_ref[10:]

        return (50 * pos_err.transpose() @ pos_err +
                0.1 * att_err.transpose() @ att_err +
                0.5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err)

class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, num_iterations):
        super().__init__(initial_state, model, controller, num_iterations)

    def update(self):
        q_des = jnp.array([0, 0, 0.5, 1, 0, 0, 0], dtype=jnp.float32)  # hovering position
        x_des = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0)
        # Compute the optimal input sequence
        time_start = time.time_ns()
        input_sequence = self.controller.compute_control_action(self.current_state, x_des).block_until_ready()
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[:self.model.nu]

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        self.state_traj[self.iter + 1, :] = self.current_state


if __name__ == "__main__":

    system = ModelMjx("bitcraze_crazyflie_2/cf2.xml")

    print(f"num states: {system.nx} ({system.nq}+{system.nv})")
    print(f"num inputs: {system.nu}")

    # mpc_config = ConfigMPC(0.02, 50, 0.1, num_parallel_computations=5000)
    # gen_config = ConfigGeneral("float32", jax.devices("gpu")[0])

    # solver = SbMPC(system, Objective(), mpc_config, gen_config)

    # # Setup and run the simulation
    # sim = Simulation(x_init, system, solver, 600)
    # sim.simulate()
    #
    # # Plot x-y position of the robot
    # plt.plot(sim.state_traj[:, 0], sim.state_traj[:, 1])
    # plt.show()
    # # Plot the input trajectory
    # plt.plot(sim.input_traj)
    # plt.show()