import os
from enum import Enum

import jax
import jax.numpy as jnp
from jax import Array

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

class RobotConfig:
    def __init__(self):
        self._robot_scene_path = ""
        self._nq = 7
        self._nv = 6
        self._nu = 4
        
        self._input_min = None
        self._input_max = None
        self.__init_input_limits_from_nu()

        q_init = jnp.zeros((self.nq))
        self._q_init = q_init

    def __init_input_limits_from_nu(self):
        input_min = jnp.zeros((self._nu))
        input_min = input_min.at[:].set(-jnp.inf)
        input_max = jnp.zeros((self._nu))
        input_max = input_max.at[:].set(jnp.inf)
        self._input_min = input_min
        self._input_max = input_max


    @property
    def robot_scene_path(self):
        return self._robot_scene_path

    @robot_scene_path.setter
    def robot_scene_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError
        if not os.path.exists(value):
            raise ValueError
        self._robot_scene_path = value
        self._nq = 7
        self._nv = 6
        self._nu = 4  
    
    @property
    def nq(self):
        return self._nq

    @nq.setter
    def nq(self, value: int):
        if not isinstance(value, int):
            raise ValueError
        self._nq = value
    
    @property
    def nv(self):
        return self._nv

    @nv.setter
    def nv(self, value: int):
        if not isinstance(value, int):
            raise ValueError
        self._nv = value
    
    
    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value: int):
        if not isinstance(value, int):
            raise ValueError
        self._nu = value
        self.__init_input_limits_from_nu()
    
    @property
    def q_init(self):
        return self._q_init

    @q_init.setter
    def q_init(self, value: Array):
        if not isinstance(value, Array):
            raise ValueError
        self._q_init = value


class Config:
    def __init__(self):
        self.general = {"dtype": jnp.float32, "device": jax.devices()[0],
                        "visualize": False}

        self.robot = RobotConfig()

        self.solver_dynamics = DynamicsModel.CLASSIC
        self.solver_type = Solver.MPPI

        self.sim_dynamics = DynamicsModel.MJX
        self.sim_iterations = 500

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
