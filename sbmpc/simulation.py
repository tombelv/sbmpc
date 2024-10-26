from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import time
import logging
import traceback
from typing import Callable, Tuple, Optional


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
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjModel, step_mujoco: bool = True, show_left_ui: bool = False, show_right_ui: bool = False):
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


class Simulator(ABC):
    def __init__(self, initial_state, model, controller, nq: int, num_iter=100, visualizer: Optional[Visualizer] = None):
        self.iter = 0
        self.current_state = initial_state
        self.model = model
        self.controller = controller
        self.num_iter = num_iter
        self.nq = nq

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
                            :self.nq])

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
