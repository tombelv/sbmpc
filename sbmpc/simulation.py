import numpy as np
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import time


class Simulator:
    def __init__(self, initial_state, model, controller, num_iter=100, visualize=False):
        self.iter = 0
        self.current_state = initial_state
        self.model = model
        self.controller = controller
        self.num_iter = num_iter



        if isinstance(initial_state, (np.ndarray, jnp.ndarray)):
            self.current_state_vec = lambda: self.current_state
        elif isinstance(initial_state, mjx.Data):
            self.current_state_vec = lambda: np.concatenate([self.current_state.qpos, self.current_state.qvel])
        else:
            raise ValueError("""
                        Invalid initial state.
                        """)

        self.state_traj = np.zeros((self.num_iter + 1, self.current_state_vec().size))

        self.state_traj[0, :] = self.current_state_vec()
        self.input_traj = np.zeros((self.num_iter, model.nu))

        self.visualize = visualize

        self.paused = False

    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused

    def update(self):
        pass

    def simulate(self):
        if self.visualize:
            with mujoco.viewer.launch_passive(self.model.mj_model,
                                              self.model.mj_data,
                                              show_left_ui=False,
                                              show_right_ui=False,
                                              key_callback=self.key_callback) as viewer:

                # viewer.cam.distance = 1.5
                viewer.cam.lookat = (0, 0, 0.6)
                while viewer.is_running() and self.iter < self.num_iter:
                    if not self.paused:
                        step_start = time.time()

                        self.step()
                        self.model.mj_data.qpos = self.current_state_vec()[:self.model.mj_model.nq]
                        mujoco.mj_kinematics(self.model.mj_model, self.model.mj_data)
                        viewer.sync()

                        time_until_next_step = self.controller.dt - (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
        else:
            while self.iter < self.num_iter:
                self.step()

    def step(self):
        self.update()
        self.iter += 1


