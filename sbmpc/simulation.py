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
from sbmpc.solvers import BaseObjective, RolloutGenerator, Controller
from sbmpc.obstacle_loader import ObstacleLoader
from typing import Callable, Tuple, Optional, Dict
from  sbmpc.sampler import Sampler, MPPISampler
from  sbmpc.gains import  Gains,MPPIGain



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

    @abstractmethod
    def move_obstacles(self, iter) -> None:
        pass


class MujocoVisualizer(Visualizer):
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, step_mujoco: bool = True, show_left_ui: bool = True, show_right_ui: bool = False, num_iters: int = 100):
        self.mj_data = mj_data
        self.mj_model = mj_model
        self.step_mujoco = step_mujoco
        self.viewer = mujoco.viewer.launch_passive(mj_model,
                                                   mj_data,
                                                   show_left_ui=show_left_ui,
                                                   show_right_ui=show_right_ui,
                                                   key_callback=self.key_callback)
        # viewer.set_cam_distance(1.5)
        self.set_cam_lookat((0, -0.25, 0.25))
        self.obsl = ObstacleLoader()
        self.obstacle_ref = self.obsl.get_obstacle_trajectory(num_iters, function="circle")

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
            raise ValueError("Invalid look at point. Size should be"
                             f" {expected_lookat_size}, {actual_size} given.")
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

    def move_obstacles(self, iter) -> None: # set obstacle positions in model according to precomputed trajectory
        n = self.obsl.n_obstacles
        obs_pos = self.obstacle_ref[iter-1]
        obs_pos = np.reshape(obs_pos, (n,3))
        for i in range(1,n+1): 
            self.mj_model.body_pos[i] = obs_pos[i-1]

def construct_mj_visualizer_from_model(model: BaseModel, config: settings.Config):
    mj_model, mj_data = (None, None)

    step_mujoco = True
    if isinstance(model, ModelMjx):
        mj_model = model.mj_model
        mj_data = model.mj_data
    else:
        new_system = ModelMjx(config.robot.robot_scene_path)
        mj_model = new_system.mj_model
        mj_data = new_system.mj_data

    visualizer = MujocoVisualizer(mj_model, mj_data, step_mujoco=step_mujoco, num_iters=config.sim_iterations) 
    return visualizer


class Simulator(ABC):
    def __init__(self, initial_state, model: BaseModel, rollout_gen: RolloutGenerator, sampler: Sampler, gains : Gains, config, visualizer: Optional[Visualizer] = None, obstacles:bool = True):
        self.iter = 0
        self.current_state = initial_state
        self.model = model
        self.controller = Controller(rollout_gen, sampler, gains)
        self.rollout_gen = rollout_gen
        self.num_iter = config.sim_iterations
        self.obstacles = obstacles

        if isinstance(initial_state, (np.ndarray, jnp.ndarray)):
            self.current_state_vec = lambda: self.current_state
        elif isinstance(initial_state, mjx.Data):
            if model.kinematic:
                self.current_state_vec = lambda: jnp.array(self.current_state.qpos)
            else:
                self.current_state_vec = lambda: jnp.concatenate(
                    [self.current_state.qpos, self.current_state.qvel])
        else:
            raise ValueError("""
                        Invalid initial state.
                        """)

        self.state_traj = np.zeros(
            (self.num_iter + 1, self.current_state_vec().size))

        self.state_traj[0, :] = self.current_state_vec() #[:self.model.nx]
        self.input_traj = np.zeros((self.num_iter, model.nu))
        self.visualizer = visualizer

        self.paused = False
        self.obstacles = obstacles

    def update(self):
        pass

    def post_update(self, data=None):
        """
        This method can be overridden to perform any post-update actions.
        It is called after the update method in each simulation step.
        """
        pass

    def simulate(self):
        if self.visualizer is not None:
            try:
                while self.visualizer.is_running() and self.iter < self.num_iter:
                    if not self.paused:
                        step_start = time.time()

                        self.step()

                        self.visualizer.set_qpos(self.current_state_vec()[
                            :self.model.get_nq()])
                        
                        if self.obstacles:
                            self.visualizer.move_obstacles(self.iter)

                        time_until_next_step = self.rollout_gen.dt - \
                            (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
                self.visualizer.close()
            except Exception as err:
                tb_str = traceback.format_exc()
                logging.error("caught exception below, closing visualizer")
                logging.error(tb_str)
                self.visualizer.close()
                raise
        else:
            while self.iter < self.num_iter:
                self.step()

    def step(self):
        self.update()
        self.post_update(self)
        self.iter += 1

ROBOT_SCENE_PATH_KEY = "robot_scene_path"

class Simulation(Simulator):
    def __init__(self, initial_state, model, controller, sampler, gains, const_reference: jnp.array, config: settings.Config, visualize_params: Optional[Dict] = None, obstacles:bool = True):
        self.const_reference = const_reference
        visualizer = None
        if config.general.visualize:
            scene_path = visualize_params.get(ROBOT_SCENE_PATH_KEY, None)
            if visualize_params is None or scene_path is None:
                raise ValueError("if visualizing need to input scene path for mjx")
            visualizer = construct_mj_visualizer_from_model(model, config)

        super().__init__(initial_state, model, controller,sampler,gains, config, visualizer, obstacles)

    def update(self):
        # Compute the optimal input sequence
        time_start = time.time_ns()
        print("iteration: ", self.iter)
        print("current state: ", self.current_state_vec())
        input_sequence = self.controller.command(self.current_state_vec(), self.const_reference, num_steps=1).block_until_ready()
        ctrl = input_sequence[0, :].block_until_ready()
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate_sim(self.current_state, ctrl, self.rollout_gen.dt)
        self.state_traj[self.iter + 1,  :] = self.current_state_vec() #[:self.model.nx] # set only qpos and qvel


def build_custom_model(custom_dynamics_fn: Callable, nq: int, nv: int, nu: int, input_min: jnp.array, input_max: jnp.array,
                        q_init: jnp.array, integrator_type: str ="si_euler", obstacle_loader: ObstacleLoader = None) -> Tuple[BaseModel, jnp.array, jnp.array]:
    system = Model(custom_dynamics_fn, nq=nq, nv=nv, nu=nu, input_bounds=[input_min, input_max], integrator_type=integrator_type)
    x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
    state_init = x_init
    return system, x_init, state_init


def build_mjx_model(config) -> Tuple[BaseModel, jnp.array, jnp.array]:
    system = ModelMjx(config.robot.robot_scene_path, kinematic=config.robot.mjx_kinematic)
    system.set_qpos(config.robot.q_init)
    q_init = system.data.qpos
    if not config.robot.mjx_kinematic:
        x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
    else:
        x_init = q_init
    state_init = system.data
    return system, x_init, state_init

def build_model_from_config(model_type: settings.DynamicsModel, config: settings.Config, custom_dynamics_fn: Optional[Callable] = None, obstacle_loader: ObstacleLoader = True):
    if model_type == settings.DynamicsModel.CUSTOM:
        if custom_dynamics_fn is None:
            raise ValueError("for classic dynamics model, a custom dynamics function must be passed. See examples.")
        nq = config.robot.nq
        nv = config.robot.nv
        nu = config.robot.nu
        input_min = config.robot.input_min
        input_max = config.robot.input_max
        q_init = config.robot.q_init
        integrator_type = config.general.integrator_type
        return build_custom_model(custom_dynamics_fn, nq, nv, nu, input_min, input_max, q_init, integrator_type, obstacle_loader)
    elif model_type == settings.DynamicsModel.MJX:
        return build_mjx_model(config)
    else:
        raise NotImplementedError

def build_model_and_solver(config: settings.Config, objective: BaseObjective, custom_dynamics_fn: Optional[Callable] = None):
    if config.solver_type != settings.Solver.MPPI:
        raise NotImplementedError
    solver_dynamics_model_setting = config.solver_dynamics
    solver_dynamics_model, solver_x_init, sim_state_init = build_model_from_config(solver_dynamics_model_setting, config, custom_dynamics_fn)
    rollout_generator = RolloutGenerator(solver_dynamics_model, objective, config)
    sampler = MPPISampler(config)
    gains = MPPIGain(config)
    return solver_dynamics_model, Controller(rollout_generator, sampler, gains)

def build_all(config: settings.Config, objective: BaseObjective,
              reference: jnp.array,
              custom_dynamics_fn: Optional[Callable] = None, 
              obstacles: bool = True):
    system, x_init, state_init = (None, None, None)
    solver_dynamics_model_setting = config.solver_dynamics
    sim_dynamics_model_setting = config.sim_dynamics

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

    if config.solver_type != settings.Solver.MPPI:
        raise NotImplementedError
    # initialize all the controller components
    # here we need to add a paramter in the config to manage the different version of sampler and gains
    rollout_generator = RolloutGenerator(solver_dynamics_model, objective, config)
    sampler = MPPISampler(config)
    gains = MPPIGain(config)
    visualize = config.general.visualize
    visualizer_params = {ROBOT_SCENE_PATH_KEY: config.robot.robot_scene_path}

    # Setup and run the simulation
    num_iterations = config.sim_iterations
    sim = Simulation(sim_state_init, sim_dynamics_model, rollout_generator, sampler, gains, reference, config, visualizer_params, obstacles)
    
    # dummy for jitting
    input_sequence = sim.controller.command(solver_x_init, reference, False).block_until_ready()
    
    return sim

