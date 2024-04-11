import numpy as np

class Simulator:
    def __init__(self, initial_state, model, controller, num_iter=100):
        self.iter = 0
        self.current_state = initial_state
        self.model = model
        self.controller = controller
        self.num_iter = num_iter

        self.state_traj = np.zeros((self.num_iter + 1, model.nx))
        self.state_traj[0, :] = self.current_state
        self.input_traj = np.zeros((self.num_iter, model.nu))

    def pre_update(self):
        pass

    def update(self):
        pass

    def post_update(self):
        pass

    def simulate(self):
        while self.iter < self.num_iter:
            self.step()

    def step(self):
        self.pre_update()
        self.update()
        self.post_update()
        self.iter += 1


