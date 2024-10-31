from enum import Enum

import jax
import jax.numpy as jnp

# Deprecated, not used now
class ConfigGeneral:
    def __init__(self, dtype_general, device: jax.Device):
        self.dtype_general = dtype_general
        self.device = device

# Deprecated, not used now
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

SOLVER_DYNAMICS_MODEL_KEY = "dynamics"
SOLVER_TYPE_KEY = "solver_type"
SIMULATION_DYNAMICS_MODEL_KEY = "dynamics"
SIMULATION_NUM_ITERATIONS_KEY = "num_iterations"
ROBOT_SCENE_PATH_KEY = "scene_path"
ROBOT_NQ_KEY = "nq"
ROBOT_NU_KEY = "nu"
ROBOT_NV_KEY = "nv"
ROBOT_INPUT_MIN_KEY = "input_min"
ROBOT_INPUT_MAX_KEY = "input_max"
ROBOT_Q_INIT_KEY = "q_init"

class DynamicsModel(Enum):
    CLASSIC = 0
    MJX = 1

class Solver(Enum):
    MPPI = 0

class Config:
    def __init__(self):
        self.general = {"dtype": jnp.float32, "device": jax.devices()[0],
                        "visualize": False}

        self.robot = {ROBOT_SCENE_PATH_KEY: "", ROBOT_NQ_KEY: 7, ROBOT_NV_KEY: 6, ROBOT_NU_KEY: 4,
                      ROBOT_INPUT_MIN_KEY: None, ROBOT_INPUT_MAX_KEY: None,
                      ROBOT_Q_INIT_KEY: None}
        input_min = jnp.zeros((self.robot[ROBOT_NU_KEY]))
        input_min = input_min.at[:].set(-jnp.inf)
        input_max = jnp.zeros((self.robot[ROBOT_NU_KEY]))
        input_max = input_max.at[:].set(jnp.inf)
        self.robot[ROBOT_INPUT_MIN_KEY] = input_min
        self.robot[ROBOT_INPUT_MAX_KEY] = input_max
        q_init = jnp.zeros((self.robot[ROBOT_NQ_KEY]))
        self.robot[ROBOT_Q_INIT_KEY] = q_init

        self.solver = {SOLVER_DYNAMICS_MODEL_KEY: DynamicsModel.CLASSIC, SOLVER_TYPE_KEY: Solver.MPPI}
        self.simulation = {SIMULATION_DYNAMICS_MODEL_KEY: DynamicsModel.MJX, SIMULATION_NUM_ITERATIONS_KEY: 500}

        self.MPC = {"dt": 0.0,
                    "horizon": -1,
                    "num_parallel_computations": 1000,
                    "lambda": 1.0,
                    "std_dev_mppi": None,
                    "initial_guess": None,
                    "filter": None,
                    "gains": False,
                    "sensitivity": False,
                    "smoothing": None,
                    "augmented_reference": False,
                    "num_control_points": -1}

    def setup(self):
        """
        Implements checks (TBD) and sets variables that depend on the user provided configuration
        """
        if self.MPC["horizon"] > 0:
            if self.MPC["num_control_points"] == -1:
                self.MPC["num_control_points"] = self.MPC["horizon"]
        else:
            raise ValueError("""
            Control horizon not set or not integer type
            Set its value using
            self.MPC["horizon"] = [INT>0]
                        """)
