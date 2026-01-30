import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

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
    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    q0 = mj_model.key_qpos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY.value, "home")]

    model_config = ModelConfig(
        dynamics_model=DynamicsModel.MJX,
        scene_path=SCENE_PATH,
        mjx_kinematic=True,
        q_init=jnp.array(q0),
    )

    controller_config = ControllerConfig(
        dt=0.02,
        horizon=50,
        num_samples=500,
        lambda_inv=100.0,
        std_dev=0.2*jnp.ones(mj_model.nu),
        smoothing="Spline",
        num_control_points=5,
        use_gains=False,
        device=jax.devices()[0],
        dtype=jnp.float32
    )

    simulation_config = SimulationConfig(
        dt=0.02,
        num_iterations=1000,
        visualize=False,
    )

    ee_des = jnp.array([-0.5, -0.5, 0.3], dtype=jnp.float32)

    model, _ = create_model(model_config)
    objective = Objective(model)
    controller = create_controller(controller_config, model, objective)
    sim = create_simulation(model, controller, simulation_config, Reference(ee_des))

    sim.run()


