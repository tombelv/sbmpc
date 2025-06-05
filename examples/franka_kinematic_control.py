import time
import os

import jax
import jax.numpy as jnp

from sbmpc import ModelMjx, BaseObjective
from sbmpc.settings import Config, RobotConfig, DynamicsModel
from sbmpc.simulation import build_all


import mujoco.mjx as mjx
import mujoco

os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
    )

SCENE_PATH = "examples/franka_emika_panda/scene.xml"



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


def post_update(sim):
    current_ee_pos = sim.controller.objective.compute_ee_pos(sim.current_state_vec())
    print(f"Current EE position: {current_ee_pos}")


if __name__ == "__main__":


    system = ModelMjx(SCENE_PATH, kinematic=True)
    q0 = system.mj_model.key_qpos[mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_KEY.value, "home")]

    robot_config = RobotConfig()

    robot_config.robot_scene_path = SCENE_PATH
    robot_config.mjx_kinematic = True
    robot_config.nu = system.mj_model.nu
    robot_config.nq = system.mj_model.nq
    robot_config.q_init = jnp.array(q0)

    config = Config(robot_config)
    config.general.visualize = True
    config.MPC.dt = 0.02
    config.MPC.horizon = 50
    config.MPC.std_dev_mppi = 0.2*jnp.ones(robot_config.nu)
    config.MPC.num_parallel_computations = 500
    config.MPC.lambda_mpc = 100.0
    config.MPC.smoothing = "Spline"
    config.MPC.num_control_points = 5

    config.solver_dynamics = DynamicsModel.MJX
    config.sim_dynamics = DynamicsModel.MJX

    config.sim_iterations = 1000

    # Reference for the end-effector position
    ee_des = jnp.array([-0.5, -0.5, 0.3], dtype=jnp.float32)

    objective = Objective(system)
    sim = build_all(
        config,
        objective,
        ee_des,
        custom_dynamics_fn=None,
        obstacles=False
    )

    sim.post_update = post_update

    sim.simulate()


