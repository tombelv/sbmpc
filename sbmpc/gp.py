import gpjax as gpx
from sbmpc.solvers import SamplingBasedMPC
from sbmpc.simulation import build_model_from_config
from examples.quadrotor import quadrotor_dynamics, Objective as quad_obj
import sbmpc.settings as settings

import jax.numpy as jnp
import numpy as np

MASS = 0.027
GRAVITY = 9.81
INPUT_HOVER = jnp.array([MASS*GRAVITY, 0., 0., 0.], dtype=jnp.float32)

class DataSet():
    def __init__(self, num_samples):  # only want to build dataset once? don't call each time
        self.num_samples = num_samples
        initial_states, control_variables, constraint_violation = self.get_quadrotor_rollouts()  # (1000,13) (1000, 25, 4) (1000,)
        inputs = np.asfarray(initial_states)
        outputs = np.asfarray(np.reshape(constraint_violation,(constraint_violation.shape[0],1)))
        self.data = gpx.Dataset(X=inputs, y=outputs)
        

    def get_quadrotor_rollouts(self):
        robot_config = settings.RobotConfig()
        
        quadrotor_config = {"nq": 7, "nv" : 6, "nu" : 4, "robot_scene_path" : "examples/bitcraze_crazyflie_2/scene.xml", "input_min" : jnp.array([0, -2.5, -2.5, -2]), "input_max" : jnp.array([1, 2.5, 2.5, 2]), "q_init" : jnp.array([0., 0., 0., 1., 0., 0., 0.], dtype=jnp.float32) }
        quadrotor_MPC_config = {"dt" : 0.02, "horizon" : 25, "std_dev_mppi" : 0.2*jnp.array([0.1, 0.1, 0.1, 0.05]), "num_parallel_computations" : 2000, "initial_guess" : INPUT_HOVER, "lambda_mpc" :50.0, "smoothing" : "Spline","num_control_points" : 5, "gains" : False }
        
        for item in quadrotor_config.keys():
            robot_config.__setattr__(item, quadrotor_config[item])

        self.config = settings.Config(robot_config)
        for item in quadrotor_MPC_config.keys():
            self.config.__setattr__(item, quadrotor_MPC_config[item])
        self.config.general.visualize = True
        self.config.solver_dynamics = settings.DynamicsModel.CUSTOM
        self.config.sim_dynamics = settings.DynamicsModel.MJX

        q_des = jnp.array([0.5, 0.5, 0.5, 1., 0., 0., 0.], dtype=jnp.float32)  
        x_des = jnp.concatenate([q_des, jnp.zeros(robot_config.nv, dtype=jnp.float32)], axis=0)
        reference = jnp.concatenate((x_des, INPUT_HOVER))
        inital_states, control_variables, constraint_violation = self.build_solver(reference)[0] # only one set of rollouts for now

        return inital_states, control_variables, constraint_violation

    
    def build_solver(self, reference):
        config = self.config
        system, x_init, state_init = (None, None, None)
        solver_dynamics_model_setting = config.solver_dynamics
        sim_dynamics_model_setting = config.sim_dynamics

        solver_dynamics_model, sim_dynamics_model = (None, None)
        solver_x_init, sim_state_init = (None, None)
        if solver_dynamics_model_setting == sim_dynamics_model_setting:
            system, solver_x_init, sim_state_init = build_model_from_config(solver_dynamics_model_setting, config, quadrotor_dynamics)
            solver_dynamics_model = system
        else:
            system, solver_x_init, _ = build_model_from_config(solver_dynamics_model_setting, config, quadrotor_dynamics)
            solver_dynamics_model = system
            if config.solver_type != settings.Solver.MPPI:
                raise NotImplementedError

        solver = SamplingBasedMPC(solver_dynamics_model, quad_obj, config)
        return solver.get_rollouts(solver_x_init, reference, False)
                

class GaussianProcessSampling():
    def __init__(self):
        dataset = DataSet(num_samples=1).data

        # starter code -> https://docs.jaxgaussianprocesses.com/_examples/regression/#constructing-the-posterior

        mean = gpx.mean_functions.Zero()
        kernel = gpx.kernels.RBF()
        prior = gpx.gps.Prior(mean_function = mean, kernel = kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=100)
        posterior = prior * likelihood

        # some visualization would be nice

    """
    -> Define prior
    -> Construct posterior
    -> Can set parameter state
    -> Predict/test on new data
    """

if __name__ == "__main__":
    gp = GaussianProcessSampling()
