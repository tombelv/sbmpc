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

os.environ['CUDA_VISIBLE_DEVICES'] = ''

SCENE_PATH = "examples/half_cheetah/half_cheetah.xml"


class Objective(BaseObjective):
    def __init__(self, model):
        super().__init__(model)
        self.object_site_id = mjx.name2id(self.robot_model.model, mujoco.mjtObj.mjOBJ_BODY.value, "head")


    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        com_pos = state[0:3]
        posture_height_cost = ((com_pos[1]-reference[1])**2).sum()
        posture_pitch_cost = ((com_pos[2]-reference[2])**2).sum()
        posture_cost = posture_height_cost + posture_pitch_cost
        vel_cost = 10*((state[9]-reference[0])**2).sum()  # desired forward velocity
        return posture_cost + vel_cost + 0.01*inputs.transpose() @ inputs

    def final_cost(self, state, reference):
        com_pos = state[0:3]
        posture_height_cost = ((com_pos[1]-reference[1])**2).sum()
        posture_pitch_cost = ((com_pos[2]-reference[2])**2).sum()
        posture_cost = posture_height_cost + posture_pitch_cost
        vel_cost = ((state[9]-reference[0])**2).sum()  # desired forward velocity
        return 100*(posture_cost + vel_cost)


def post_update(sim):
    print(f"Current COM position: {sim.current_state_vec()[0:2]}")


if __name__ == "__main__":


    system = ModelMjx(SCENE_PATH, kinematic=False)
    q0 = system.mj_model.key_qpos[mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_KEY.value, "home")]

    robot_config = RobotConfig()

    robot_config.robot_scene_path = SCENE_PATH
    robot_config.mjx_kinematic = False
    robot_config.nu = system.mj_model.nu
    robot_config.nq = system.mj_model.nq
    robot_config.nv = system.mj_model.nv
    robot_config.q_init = jnp.array(q0)

    config = Config(robot_config)
    config.general.visualize = False
    config.MPC.dt = 0.02
    config.MPC.horizon = 30
    config.MPC.std_dev_mppi = 0.2*jnp.ones(robot_config.nu)
    config.MPC.num_parallel_computations = 1500
    config.MPC.lambda_mpc = 100.0
    #config.MPC.smoothing = "Spline"
    #config.MPC.num_control_points = 5
    config.MPC.num_control_points = config.MPC.horizon
    config.MPC.gains = True

    config.solver_dynamics = DynamicsModel.MJX
    config.sim_dynamics = DynamicsModel.MJX

    config.sim_iterations = 500

    config.sim.dt = 0.005

    # Reference for the end-effector position
    final_com_pos = jnp.array([0.8, 0., 0.], dtype=jnp.float32)

    objective = Objective(system)
    sim = build_all(
        config,
        objective,
        final_com_pos,
        custom_dynamics_fn=None,
        obstacles=False
    )

    sim.post_update = post_update

    sim.simulate()

    import matplotlib.pyplot as plt
    time_vect = config.MPC.dt*jnp.arange(sim.state_traj.shape[0])
    plt.plot(time_vect, sim.state_traj[:, 9])
    plt.legend(["v_x"])
    plt.grid()
    plt.show()

    plt.show()
