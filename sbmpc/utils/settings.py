import jax


class ConfigGeneral:
    def __init__(self, dtype_general, device: jax.Device):
        self.dtype_general = dtype_general
        self.device = device


class ConfigMPC:
    def __init__(self, dt: float, horizon: int, sigma_mppi, num_parallel_computations: int = 10000, initial_guess = None):
        self.dt = dt
        self.horizon = horizon
        self.num_parallel_computations = num_parallel_computations

        self.initial_guess = initial_guess

        self.sigma_mppi = sigma_mppi


