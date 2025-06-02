import os
from enum import Enum

import jax
import jax.numpy as jnp
from jax import Array, Device

MODEL_PARAMETRIC_INTEGRATOR_TYPES = ["si_euler", "euler", "rk4", "custom_discrete"]


class DynamicsModel(Enum):
    CUSTOM = 0
    MJX = 1


class Solver(Enum):
    MPPI = 0


class RobotConfig:
    def __init__(self):
        self._robot_scene_path = ""
        self._mjx_kinematic = False
        self._nq = 0
        self._nv = 0
        self._nu = None
        self._nx = 0

        self._input_min = None
        self._input_max = None
        self._q_init = None

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
            raise ValueError("str type is expected")
        if not os.path.exists(value):
            raise FileNotFoundError
        self._robot_scene_path = value


    @property
    def mjx_kinematic(self):
        return self._mjx_kinematic

    @mjx_kinematic.setter
    def mjx_kinematic(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("bool type is expected")
        self._mjx_kinematic = value
    
    @property
    def nq(self):
        return self._nq

    @nq.setter
    def nq(self, value: int):
        if not isinstance(value, int):
            raise ValueError("int type is expected")
        self._nq = value
        if self._nv is not None:
            self._nx = self._nq + self._nv
    
    @property
    def nv(self):
        return self._nv

    @nv.setter
    def nv(self, value: int):
        if not isinstance(value, int):
            raise ValueError("int type is expected")
        self._nv = value
        if self._nq is not None:
            self._nx = self._nq + self._nv
    
    @property
    def nx(self):
        return self._nx
    
    @property
    def input_min(self):
        return self._input_min
    
    @input_min.setter
    def input_min(self, value: Array):
        if not isinstance(value, Array):
            raise ValueError("jax Array type is expected")
        if len(value) != self._nu:
            raise ValueError("length must match nu")
        self._input_min = value
    
    @property
    def input_max(self):
        return self._input_max
    
    @input_max.setter
    def input_max(self, value: Array):
        if not isinstance(value, Array):
            raise ValueError("jax Array type is expected")
        if len(value) != self._nu:
            raise ValueError("length must match nu")
        self._input_max = value
    


    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value: int):
        if not isinstance(value, int):
            raise ValueError("int type is expected")
        self._nu = value
        self.__init_input_limits_from_nu()
    
    @property
    def q_init(self):
        return self._q_init

    @q_init.setter
    def q_init(self, value: Array):
        if not isinstance(value, Array):
            raise ValueError("jax Array type is expected")
        self._q_init = value


class MPCConfig:
    def __init__(self, config: RobotConfig):
        self._dt = 0.0
        self._horizon = 1
        self._num_parallel_computations = 1000
        self._lambda_mpc = 1.0
        self._filter = None
        self._gains = False
        self._sensitivity = False
        self._smoothing = None
        self._augmented_reference = None
        self._num_control_points = 0

        self._std_dev_mppi = jnp.zeros(config.nu)
        self._initial_guess = jnp.zeros(config.nu)
        

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value: float):
        if not isinstance(value, float):
            raise ValueError("float type is expected")
        self._dt = value

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value: int):
        if not isinstance(value, int):
            raise ValueError("int type is expected")
        if not value > 0:
            raise ValueError("must be greater than zero")
        self._horizon = value

    @property
    def num_parallel_computations(self):
        return self._num_parallel_computations

    @num_parallel_computations.setter
    def num_parallel_computations(self, value: int):
        if not isinstance(value, int):
            raise ValueError("int type is expected")
        self._num_parallel_computations = value
    
    @property
    def lambda_mpc(self):
        return self._lambda_mpc

    @lambda_mpc.setter
    def lambda_mpc(self, value: float):
        if not isinstance(value, float):
            raise ValueError("float type is expected")
        self._lambda_mpc = value

    @property
    def std_dev_mppi(self):
        return self._std_dev_mppi

    @std_dev_mppi.setter
    def std_dev_mppi(self, value: Array):
        if not isinstance(value, Array):
            raise ValueError("jax Array type is expected")
        if len(value) != self._std_dev_mppi.size:
            raise ValueError("length must match nu")
        self._std_dev_mppi = value

    @property
    def initial_guess(self):
        return self._initial_guess

    @initial_guess.setter
    def initial_guess(self, value: Array):
        if ((not isinstance(value, Array)) and (value is not None)):
            raise ValueError("jax Array type or None is expected")
        if value is not None and len(value) != self._initial_guess.size:
            raise ValueError("length must match nu")
        self._initial_guess = value

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        self._filter = value

    @property
    def gains(self):
        return self._gains

    @gains.setter
    def gains(self, value):
        if not isinstance(value, bool):
            raise ValueError("bool type is expected")
        self._gains = value

    @property
    def sensitivity(self):
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        if not isinstance(value, bool):
            raise ValueError("bool type is expected")
        self._sensitivity = value

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        if not isinstance(value, str) and value != None:
            raise ValueError("str type is expected")
        supported_smoothing = ["Spline", None]
        if value not in supported_smoothing:
            raise ValueError(f"smoothing not supported. Choose from {supported_smoothing}")
        self._smoothing = value

    @property
    def augmented_reference(self):
        return self._augmented_reference

    @augmented_reference.setter
    def augmented_reference(self, value):
        self._augmented_reference = value

    @property
    def num_control_points(self):
        return self._num_control_points

    @num_control_points.setter
    def num_control_points(self, value):
        if not isinstance(value, int):
            raise ValueError("int type is expected")
        if not value > 0:
            raise ValueError("must be greater than zero")
        self._num_control_points = value

class GeneralConfig:
    def __init__(self):
        self._dtype = jnp.float32
        self._device = jax.devices()[0]
        self._visualize = False
        self._integrator_type = "si_euler"

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if not isinstance(value, type):
            raise ValueError("'type' type is expected")
        self._dtype = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if not isinstance(value, Device):
            raise ValueError("'Device' type is expected")
        self._device = value

    @property
    def visualize(self):
        return self._visualize

    @visualize.setter
    def visualize(self, value):
        if not isinstance(value, bool):
            raise ValueError("bool type is expected")
        self._visualize = value

    @property
    def integrator_type(self):
        return self._integrator_type

    @integrator_type.setter
    def integrator_type(self, value):
        if not isinstance(value, str):
            raise ValueError("str type is expected")
        if not value in MODEL_PARAMETRIC_INTEGRATOR_TYPES:
            raise ValueError(f"value must be in list of types: {MODEL_PARAMETRIC_INTEGRATOR_TYPES}")
        self._integrator_type = value

class Config:
    def __init__(self, robot_config: RobotConfig):
        self.general = GeneralConfig()

        self.robot = robot_config

        self._solver_dynamics = DynamicsModel.CUSTOM
        self.solver_type = Solver.MPPI

        self.sim_dynamics = DynamicsModel.CUSTOM
        self.sim_iterations = 500

        self.MPC = MPCConfig(robot_config)

    @property
    def solver_dynamics(self):
        return self._solver_dynamics
    
    @solver_dynamics.setter
    def solver_dynamics(self, value: DynamicsModel):
        if not isinstance(value, DynamicsModel):
            raise ValueError("DynamicsModel type is expected")
        self._solver_dynamics = value


