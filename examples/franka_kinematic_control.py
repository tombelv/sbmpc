import time, os

import jax
import jax.numpy as jnp

os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
    )

import matplotlib.pyplot as plt

from sbmpc.model import Model, ModelMjx
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
from sbmpc.utils.geometry import skew, quat_product, quat2rotm, quat_inverse
import sbmpc.utils.simulation as simulation

import mujoco.mjx as mjx
import mujoco
import mujoco.viewer

class Objective(BaseObjective):
    def __init__(self, model):
        super().__init__(model)
        self.ee_site_id = mjx.name2id(self.robot_model.model, mujoco.mjtObj.mjOBJ_BODY.value, "attachment")
        self.compute_ee_pos = jax.jit(self._compute_ee_pos)

    def _compute_ee_pos(self, configuration):
        mjx_data = self.robot_model.data
        mjx_data = mjx_data.replace(qpos=configuration)

        mjx_data = mjx.kinematics(self.robot_model.model, mjx_data)

        return mjx_data.xpos[self.ee_site_id]

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        ee_pos = self.compute_ee_pos(state)
        return ((ee_pos - reference[:3])**2).sum() + 0.1*inputs.transpose() @ inputs

    def final_cost(self, state, reference):
        ee_pos = self.compute_ee_pos(state)
        return 100*((ee_pos - reference[:3]) ** 2).sum()


class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, num_iterations, visualize):
        super().__init__(initial_state, model, controller, num_iterations, visualize)

    def update(self):
        ee_des = jnp.array([-0.5, 0.4, 0.3], dtype=jnp.float32)  # hovering position
        reference = ee_des
        # Compute the optimal input sequence
        input_sequence = self.controller.compute_control_action(self.current_state_vec(), reference, num_steps=1)
        ctrl = input_sequence[:self.model.nu]

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        self.state_traj[self.iter + 1, :] = self.current_state_vec()

        print("EE position = ", self.controller.objective.compute_ee_pos(self.current_state))



if __name__ == "__main__":

    mpc_config = ConfigMPC(dt=0.02,  # Sampling time
                           horizon=25,  # control horizon
                           std_dev_mppi=0.2*jnp.ones(7),  # input std dev
                           num_parallel_computations=2000)
    gen_config = ConfigGeneral("float32", jax.devices("gpu")[0])

    system = ModelMjx("franka_emika_panda/scene.xml", kinematic=True)
    system.set_qpos(system.mj_model.key_qpos[mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_KEY.value, "home")])
    q_init = system.data.qpos
    print("Initial configuration = ", q_init)

    solver = SbMPC(system, Objective(system), mpc_config, gen_config)

    # dummy for jitting
    solver.compute_control_action(q_init, jnp.zeros(3)).block_until_ready()

    # Setup and run the simulation
    sim = Simulation(q_init, system, solver, 5000, visualize=True)
    sim.simulate()


