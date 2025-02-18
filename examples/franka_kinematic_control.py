import time
import os

import jax
import jax.numpy as jnp

from sbmpc import ModelMjx, SamplingBasedMPC, BaseObjective
from sbmpc.settings import Config, RobotConfig
from sbmpc.simulation import Simulator, construct_mj_visualizer_from_model


import mujoco.mjx as mjx
import mujoco

os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
    )

SCENE_PATH = "franka_emika_panda/scene.xml"



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


class Simulation(Simulator):
    def __init__(self, initial_state, model, controller, num_iterations: int, visualize: bool = False):
        visualizer = None
        if visualize:
            visualizer = construct_mj_visualizer_from_model(model, SCENE_PATH, 5000)
        super().__init__(initial_state, model, controller, num_iterations, visualizer)

    def run_kinematics(self):
        mujoco.mj_kinematics(
                            self.model.mj_model, self.model.mj_data)

    def update(self):
        ee_des = jnp.array([-0.5, 0.4, 0.3], dtype=jnp.float32)  # hovering position
        reference = ee_des
        # Compute the optimal input sequence
        input_sequence = self.controller.command(self.current_state_vec(), reference, num_steps=1)
        ctrl = input_sequence[0]

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        self.state_traj[self.iter + 1, :] = self.current_state_vec()
        self.run_kinematics()

        print("EE position = ", self.controller.objective.compute_ee_pos(self.current_state))


if __name__ == "__main__":

    robot_config = RobotConfig()

    robot_config.robot_scene_path = SCENE_PATH
    robot_config.mjx_kinematic = True
    robot_config.nu = 7

    config = Config(robot_config)
    config.general.visualize = True
    config.MPC.dt = 0.02
    config.MPC.horizon = 50
    config.MPC.std_dev_mppi = 0.2*jnp.ones(robot_config.nu)
    config.MPC.num_parallel_computations = 1000
    config.MPC.lambda_mpc = 100.0
    config.MPC.smoothing = "Spline"
    config.MPC.num_control_points = 5
    config.MPC.gains = False
    config.MPC.sensitivity = False

    # system, x_init, state_init = build_model_from_config(DynamicsModel.MJX, config)
    system = ModelMjx(SCENE_PATH, kinematic=True)
    system.set_qpos(system.mj_model.key_qpos[mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_KEY.value, "home")])
    q_init = system.data.qpos
    print("Initial configuration = ", q_init)

    x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
    state_init = system.data

    objective = Objective(system)
    solver = SamplingBasedMPC(system, objective, config)

    # dummy for jitting
    ref = jnp.zeros(3, dtype=jnp.float32)
    solver.command(q_init, ref).block_until_ready()

    # Setup and run the simulation
    sim = Simulation(q_init, system, solver, 5000, visualize=config.general.visualize)
    sim.simulate()


