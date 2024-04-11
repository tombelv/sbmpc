

class Simulator:
    def __init__(self, initial_state, model, controller, num_iterations=100):
        self.iter = 0
        self.current_state = initial_state
        self.model = model
        self.controller = controller
        self.num_iterations = num_iterations

    def pre_update(self):
        pass

    def update(self):
        pass

    def post_update(self):
        pass

    def simulate(self):
        while self.iter < self.num_iterations:
            self.step()

    def step(self):
        self.pre_update()
        self.update()
        self.post_update()
        self.iter += 1


