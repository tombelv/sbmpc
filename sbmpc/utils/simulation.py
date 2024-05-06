import numpy as np
import jax.numpy as jnp
import mujoco.mjx as mjx


class Simulator:
    def __init__(self, initial_state, model, controller, num_iter=100):
        self.iter = 0
        self.current_state = initial_state
        self.model = model
        self.controller = controller
        self.num_iter = num_iter

        self.state_traj = np.zeros((self.num_iter + 1, model.nx))

        if isinstance(initial_state, (np.ndarray, jnp.ndarray)):
            self.state_traj[0, :] = self.current_state
        elif isinstance(initial_state, mjx.Data):
            self.state_traj[0, :] = np.concatenate([self.current_state.qpos, self.current_state.qvel])
        else:
            raise ValueError("""
                        Invalid initial state.
                        """)

        self.input_traj = np.zeros((self.num_iter, model.nu))

    def update(self):
        pass

    def simulate(self):
        while self.iter < self.num_iter:
            self.step()

    def step(self):
        self.update()
        self.iter += 1


