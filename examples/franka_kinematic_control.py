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




if __name__ == "__main__":

    robot_config = RobotConfig()

    robot_config.robot_scene_path = SCENE_PATH
    robot_config.mjx_kinematic = True
    robot_config.nu = 7
    robot_config.nq = 7

    config = Config(robot_config)
    config.general.visualize = False
    config.MPC.dt = 0.02
    config.MPC.horizon = 50
    config.MPC.std_dev_mppi = 0.2*jnp.ones(robot_config.nu)
    config.MPC.num_parallel_computations = 10000
    config.MPC.lambda_mpc = 10.0
    config.MPC.smoothing = "Spline"
    config.MPC.num_control_points = 5


    config.solver_dynamics = DynamicsModel.MJX
    config.sim_dynamics = DynamicsModel.MJX

    system = ModelMjx(SCENE_PATH, kinematic=True)
    system.set_qpos(system.mj_model.key_qpos[mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_KEY.value, "home")])
    robot_config.q_init = jnp.array(system.data.qpos)
    print("Initial configuration = ", system.data.qpos)

    # Reference for the end-effector position
    ee_des = jnp.array([-0.5, 0.4, 0.3], dtype=jnp.float32)

    # x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
    # state_init = system.data

    objective = Objective(system)
    sim = build_all(
        config,
        objective,
        ee_des,
        custom_dynamics_fn=None,
        obstacles=False
    )

    sim.simulate()


