import jax
import jax.numpy as jnp


class ConfigGeneral:
    def __init__(self, dtype_general, device: jax.Device):
        self.dtype_general = dtype_general
        self.device = device


class ConfigMPC:
    def __init__(self, dt: float, horizon: int, std_dev_mppi: jnp.array, num_parallel_computations: int = 10000,
                 initial_guess=None):
        self.dt = dt
        self.horizon = horizon
        self.num_parallel_computations = num_parallel_computations

        if initial_guess is None:
            self.initial_guess = 0.0 * std_dev_mppi
        else:
            self.initial_guess = initial_guess

        self.std_dev_mppi = std_dev_mppi

        self.filter = None


class Config:
    def __init__(self):
        self.general = {"dtype": jnp.float32, "device": jax.devices()[0]}
        self.MPC = {"dt": 0.0,
                    "horizon": 1,
                    "num_parallel_computations": 1000,
                    "std_dev_mppi": None,
                    "initial_guess": None,
                    "filter": None}
