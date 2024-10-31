from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import time
import logging
import traceback
from sbmpc.model import BaseModel, Model, ModelMjx
import sbmpc.settings as settings
from sbmpc.solvers import BaseObjective, SamplingBasedMPC
from typing import Callable, Tuple, Optional, Dict


class Visualizer(ABC):
    def __init__(self):
        self.paused = False
        
    def toggle_paused(self):
        self.paused != self.paused
    
    def get_paused(self):
        return self.paused

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def set_cam_lookat(self, lookat_point: Tuple[float]) -> None:
        pass

    @abstractmethod
    def set_cam_distance(self, distance: float) -> None:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def set_qpos(self, qpos) -> None:
        pass


class MujocoVisualizer(Visualizer):
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, step_mujoco: bool = True, show_left_ui: bool = False, show_right_ui: bool = False):
        self.mj_data = mj_data
        self.mj_model = mj_model
        self.step_mujoco = step_mujoco
        self.viewer = mujoco.viewer.launch_passive(mj_model,
                                                   mj_data,
                                                   show_left_ui=show_left_ui,
                                                   show_right_ui=show_right_ui,
                                                   key_callback=self.key_callback)

    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.toggle_paused()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.viewer.__exit__(exc_type, exc_val, exc_tb)

    def set_cam_lookat(self, lookat_point: Tuple) -> None:
        expected_lookat_size = 3
        actual_size = len(lookat_point)
        if actual_size != expected_lookat_size:
            raise ValueError(f"Invalid look at point. Size should be {
                             expected_lookat_size}, {actual_size} given.")
        self.viewer.cam.lookat = lookat_point

    def set_cam_distance(self, distance: float) -> None:
        self.viewer.cam.distance = distance

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def set_qpos(self, qpos) -> None:
        if self.step_mujoco:
            self.mj_data.qpos = qpos
            mujoco.mj_fwdPosition(self.mj_model, self.mj_data)
        self.viewer.sync()

def construct_mj_visualizer_from_model(model: BaseModel, scene_path: str):
    mj_model, mj_data = (None, None)
    step_mujoco = True
    if isinstance(model, ModelMjx):
        mj_model = model.mj_model
        mj_data = model.mj_data
    else:
        new_system = ModelMjx(scene_path)
        mj_model = new_system.mj_model
        mj_data = new_system.mj_data
    visualizer = MujocoVisualizer(mj_model, mj_data, step_mujoco=step_mujoco)
    return visualizer


class Simulator(ABC):
    def __init__(self, initial_state, model: BaseModel, controller, num_iter=100, visualizer: Optional[Visualizer] = None):
        self.iter = 0
        self.current_state = initial_state
        self.model = model
        self.controller = controller
        self.num_iter = num_iter

        if isinstance(initial_state, (np.ndarray, jnp.ndarray)):
            self.current_state_vec = lambda: self.current_state
        elif isinstance(initial_state, mjx.Data):
            self.current_state_vec = lambda: np.concatenate(
                [self.current_state.qpos, self.current_state.qvel])
        else:
            raise ValueError("""
                        Invalid initial state.
                        """)

        self.state_traj = np.zeros(
            (self.num_iter + 1, self.current_state_vec().size))

        self.state_traj[0, :] = self.current_state_vec()
        self.input_traj = np.zeros((self.num_iter, model.nu))

        self.visualizer = visualizer

        self.paused = False

    @abstractmethod
    def update(self):
        pass

    def simulate(self):
        if self.visualizer is not None:
            try:
                # viewer.set_cam_distance(1.5)
                self.visualizer.set_cam_lookat((0, 0, 0.6))
                while self.visualizer.is_running() and self.iter < self.num_iter:
                    if not self.paused:
                        step_start = time.time()

                        self.step()

                        self.visualizer.set_qpos(self.current_state_vec()[
                            :self.model.get_nq()])

                        time_until_next_step = self.controller.dt - \
                            (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
            except Exception as err:
                tb_str = traceback.format_exc()
                logging.error("caught exception below, closing visualizer")
                logging.error(tb_str)
                self.visualizer.close()
                raise
            self.visualizer.close()
        else:
            while self.iter < self.num_iter:
                self.step()

    def step(self):
        self.update()
        self.iter += 1


class Simulation(Simulator):
    def __init__(self, initial_state, model, controller, const_reference: jnp.array, num_iterations: int, visualize: bool = True, visualize_params: Optional[Dict] = None):
        self.const_reference = const_reference
        visualizer = None
        if visualize:
            scene_path = visualize_params.get(settings.ROBOT_SCENE_PATH_KEY, None)
            if visualize_params is None or scene_path is None:
                raise ValueError("if visualizing need to input scene path for mjx")
            visualizer = construct_mj_visualizer_from_model(model, scene_path)

        super().__init__(initial_state, model, controller, num_iterations, visualizer)

    def update(self):
        # Compute the optimal input sequence
        time_start = time.time_ns()
        input_sequence = self.controller.command(self.current_state_vec(), self.const_reference, num_steps=1).block_until_ready()
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[:self.model.nu]

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        self.state_traj[self.iter + 1, :] = self.current_state_vec()


def build_classic_model(custom_dynamics_fn: Callable, nq: int, nv: int, nu: int, input_min: jnp.array, input_max: jnp.array,
                        q_init: jnp.array) -> Tuple[BaseModel, jnp.array, jnp.array]:
    system = Model(custom_dynamics_fn, nq=nq, nv=nv, nu=nu, input_bounds=[input_min, input_max])
    x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
    state_init = x_init
    return system, x_init, state_init


def build_mjx_model(scene_path: str) -> Tuple[BaseModel, jnp.array, jnp.array]:
    system = ModelMjx(scene_path)
    q_init = system.data.qpos
    x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
    state_init = system.data
    return system, x_init, state_init

def build_model_from_config(model_type: settings.DynamicsModel, config: settings.Config, custom_dynamics_fn: Optional[Callable] = None):
    if model_type == settings.DynamicsModel.CLASSIC:
        if custom_dynamics_fn is None:
            raise ValueError("for classic dynamics model, a custom dynamics function must be passed. See examples.")
        nq = config.robot[settings.ROBOT_NQ_KEY]
        nv = config.robot[settings.ROBOT_NV_KEY]
        nu = config.robot[settings.ROBOT_NU_KEY]
        input_min = config.robot[settings.ROBOT_INPUT_MIN_KEY]
        input_max = config.robot[settings.ROBOT_INPUT_MAX_KEY]
        q_init = config.robot[settings.ROBOT_Q_INIT_KEY]
        return build_classic_model(custom_dynamics_fn, nq, nv, nu, input_min, input_max, q_init)
    elif model_type == settings.DynamicsModel.MJX:
        return build_mjx_model(config.robot[settings.ROBOT_SCENE_PATH_KEY])
    else:
        raise NotImplementedError

def build_all(config: settings.Config, objective: BaseObjective,
              reference: jnp.array,
              custom_dynamics_fn: Optional[Callable] = None):
    system, x_init, state_init = (None, None, None)
    solver_dynamics_model_setting = config.solver[settings.SOLVER_DYNAMICS_MODEL_KEY]
    sim_dynamics_model_setting = config.simulation[settings.SIMULATION_DYNAMICS_MODEL_KEY]

    solver_dynamics_model, sim_dynamics_model = (None, None)
    solver_x_init, sim_state_init = (None, None)
    if solver_dynamics_model_setting == sim_dynamics_model_setting:
        system, solver_x_init, sim_state_init = build_model_from_config(solver_dynamics_model_setting, config, custom_dynamics_fn)
        solver_dynamics_model = system
        sim_dynamics_model = system
    else:
        system, solver_x_init, _ = build_model_from_config(solver_dynamics_model_setting, config, custom_dynamics_fn)
        solver_dynamics_model = system
        sim_dynamics_model, _, sim_state_init = build_model_from_config(sim_dynamics_model_setting, config, custom_dynamics_fn)

    if config.solver[settings.SOLVER_TYPE_KEY] != settings.Solver.MPPI:
        raise NotImplementedError

    solver = SamplingBasedMPC(solver_dynamics_model, objective, config)
    
    # dummy for jitting
    input_sequence = solver.command(solver_x_init, reference, False).block_until_ready()
    visualize = config.general["visualize"]
    visualizer_params = {settings.ROBOT_SCENE_PATH_KEY: config.robot[settings.ROBOT_SCENE_PATH_KEY]}

    # Setup and run the simulation
    num_iterations = config.simulation[settings.SIMULATION_NUM_ITERATIONS_KEY]
    sim = Simulation(sim_state_init, sim_dynamics_model, solver, reference, num_iterations, visualize, visualizer_params)
    return sim

