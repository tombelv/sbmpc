import jax
import jax.numpy as jnp
import gpjax as gpx

import sbmpc.settings as settings
import numpy as np
import pandas as pd
# import hmc  # TODO import manually ... pip installed version not correct
import time
import httpimport

from sbmpc.solvers import SamplingBasedMPC
from sbmpc.simulation import build_model_from_config, BgSimulator
from examples.quadrotor import quadrotor_dynamics
from examples.quadrotor_obstacles import Objective
from sbmpc.obstacle_loader import ObstacleLoader
from sklearn.metrics import mean_squared_error, r2_score # TODO - check these after optimisation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from jax.scipy.stats import norm

with httpimport.github_repo('martin-marek', 'mini-hmc-jax', ref='master'):
  import hmc

MASS = 0.027
GRAVITY = 9.81
INPUT_HOVER = jnp.array([MASS*GRAVITY, 0., 0., 0.], dtype=jnp.float32)
obsl = ObstacleLoader()
num_steps = 10

class DataSet():
    def __init__(self):
        pass

    def create(self, num_samples, make_new=False):  # num_steps = number of steps to simulate, num_samples - number of parallel computations to take from each step
        samples = None
        if make_new:
            samples = self.get_quadrotor_rollouts(num_samples) # ((num_samples * num_steps), 114)
            samples[:,-1] = np.abs(samples[:,-1]) # constraints term should be absolute TODO - check this in solvers
            np.savetxt("/home/ubuntu/sbmpc/sbmpc/quadrotor_rollouts.data", samples, fmt='%4.6f', delimiter=' ')
             
        else: 
            samples = pd.read_csv("/home/ubuntu/sbmpc/sbmpc/quadrotor_rollouts.data", header=None, delimiter=' ').values[:num_samples*num_steps]

        print(f"Samples shape => {samples.shape}")
        X = samples[:, :-1]  
        y = samples[:, -1].reshape(-1, 1)

        return X , y
  
    def get_quadrotor_rollouts(self, num_samples): 
        robot_config = settings.RobotConfig() # run simulation as usual
        robot_config.robot_scene_path = "examples/bitcraze_crazyflie_2/scene.xml"
        robot_config.nq = 7
        robot_config.nv = 6
        robot_config.nu = 4
        robot_config.input_min = jnp.array([0, -2.5, -2.5, -2])
        robot_config.input_max = jnp.array([1, 2.5, 2.5, 2])
        robot_config.q_init = jnp.array([0., 0., 0., 1., 0., 0., 0.], dtype=jnp.float32) 
        
        config = settings.Config(robot_config)
        config.general.visualize = True
        config.MPC.dt = 0.02
        config.MPC.horizon = 25
        config.MPC.std_dev_mppi = 0.2*jnp.array([0.1, 0.1, 0.1, 0.05])
        config.MPC.num_parallel_computations = 2000
        config.MPC.initial_guess = INPUT_HOVER
        config.MPC.lambda_mpc = 50.0
        config.MPC.smoothing = "Spline"
        config.MPC.num_control_points = 5
        config.MPC.gains = False
        config.solver_dynamics = settings.DynamicsModel.CUSTOM
        config.sim_dynamics = settings.DynamicsModel.MJX
        self.config = config

        q_des = jnp.array([0.5, 0.5, 0.5, 1., 0., 0., 0.], dtype=jnp.float32) 
        x_des = jnp.concatenate([q_des, jnp.zeros(robot_config.nv, dtype=jnp.float32)], axis=0)

        horizon = config.MPC.horizon+1
        traj = obsl.get_obstacle_trajectory(config.sim_iterations,"circle")[:horizon]   # TODO - include diagonal (and sine?)

        reference = jnp.concatenate((x_des, INPUT_HOVER))  
        reference = jnp.tile(reference, (horizon, 1))
        reference = jnp.concatenate([reference, traj],axis=1)

        bg_sim = self.build_solver_and_simulator(reference)

        rows = num_samples*num_steps
        initial_states_all = np.zeros((rows,13))  
        # control_vars_all = np.zeros((rows,100))
        control_vars_all = np.zeros((rows,25))
        constraint_violation_all = np.zeros((rows,1))

        lambda_ = 0.5
        for i in range(num_steps): # take first 100 rollouts for each state (currently 100 states) 
            initial_states, control_vars, constraint_violation = bg_sim.run()[0] # returns 2000 parallel rollouts for given state

            initial_states = np.array(initial_states)[:num_samples]
            # control_vars = np.reshape(control_vars[:,:,0],(2000,25,1))
            control_vars = control_vars[:,:,0]
            control_vars = np.reshape(control_vars,(control_vars.shape[0],-1))[:num_samples] 
            constraint_violation = np.reshape(constraint_violation,(constraint_violation.shape[0], 1))[:num_samples]
           
            idx = i*num_samples
            # all_samples[idx:idx+num_samples] = samples

            initial_states_all[idx:idx+num_samples] = initial_states
            control_vars_all[idx:idx+num_samples] = control_vars
            constraint_violation_all[idx:idx+num_samples] = constraint_violation 

        rollouts = np.float64(np.concatenate([initial_states_all, control_vars_all, constraint_violation_all], axis=1))
        # rollouts = np.float64(np.concatenate([initial_states_all,  constraint_violation_all], axis=1))  #[:,:5]

        return rollouts
    
    def build_solver_and_simulator(self, reference):
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
            sim_dynamics_model, _, sim_state_init = build_model_from_config(sim_dynamics_model_setting, config, quadrotor_dynamics)
            solver_dynamics_model = system
            if config.solver_type != settings.Solver.MPPI:
                raise NotImplementedError
       
        quad_obj = Objective()
        solver = SamplingBasedMPC(solver_dynamics_model, quad_obj, config)
        sim = BgSimulator(sim_state_init, sim_dynamics_model, solver, reference, num_iter=100, visualizer = None, obstacles = True)

        return sim
                

class GaussianProcessSampling(): 
    def __init__(self):
        self.key = jax.random.key(456)
        self.ds = DataSet()
        self.training_set = None
        self.test_set = None
        self.n_samples = 1999
        self.P = np.ones((self.n_samples,1,1)) 
        
        shape = (self.n_samples, 5, 4)
        self.mean = np.float32(0.467)
        self.stddev = np.reshape(0.2*jnp.array([0.1, 0.1, 0.1, 0.05]),(4,1)).T
       
        # self.mppi_gauss = (jax.random.normal(key=self.key, shape=shape) + self.mean) * self.stddev
        self.mppi_gauss = (jax.random.normal(key=self.key, shape=shape) + self.mean) 
        self.flat_mppi_gauss = jax.random.normal(key=self.key, shape=(self.n_samples,38)) + self.mean
        self.target_mean = 0
        self.target_stddev = 1
          
    def run(self):
        X, y = self.ds.create(num_samples=50)
        
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

        log_ytr = np.log(ytr)
        log_yte = np.log(yte)

        y_scaler = StandardScaler().fit(log_ytr)
        scaled_ytr = y_scaler.transform(log_ytr)
        scaled_yte = y_scaler.transform(log_yte)

        x_scaler = StandardScaler().fit(Xtr)
        scaled_Xtr = x_scaler.transform(Xtr)
        scaled_Xte = x_scaler.transform(Xte)

        # print(f"Xtr = {np.all(np.isnan(scaled_Xtr) == False)}, Xte = {np.all(np.isnan(scaled_Xte) == False)}, ytr = {np.all(np.isnan(scaled_ytr) == False)}, yte = {np.all(np.isnan(scaled_yte) == False)}")

        n_train, n_covariates = scaled_Xtr.shape
        kernel = gpx.kernels.RBF(
            active_dims=list(range(n_covariates)),
            variance=jnp.var(scaled_ytr),
            lengthscale=0.1 * jnp.ones((n_covariates,)),
            )
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_train) 
        mean = gpx.mean_functions.Zero()  
        kernel = gpx.kernels.RBF()

        prior = gpx.gps.Prior(mean_function = mean, kernel = kernel) 
        posterior = prior * likelihood
        self.training_set = gpx.Dataset(X=scaled_Xtr, y=scaled_ytr)  
        self.test_set = gpx.Dataset(X=scaled_Xte, y=scaled_yte)
    
        opt_posterior, history = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_loocv(p,d),   # vs conjugate mll 
        train_data=self.training_set,
        ) 
        # print(history)

        mock_X = jax.random.normal(key=self.key, shape= (self.n_samples,38)) # flat_mppi_gauss
        mock_Y = jax.random.normal(key=self.key, shape = (self.n_samples,)) # mock contraint terms for the input - will compute in solvers when integrated
        
        P = self.get_gp_P(opt_posterior, self.training_set, mock_X, mock_Y) 
        target_dist = self.mppi_gauss * P

        self.target_mean = jnp.mean(target_dist)
        self.target_stddev = jnp.std(target_dist)

    def get_gp_P(self, opt_posterior, training_set, X, Y):
        latent_dist = opt_posterior.predict(X, train_data=training_set) # get GP prediction for X (Y)
        predictive_dist = opt_posterior.likelihood(latent_dist) # determine how likely these samples are to fit original Gp
        
        mean = predictive_dist.mean()
        stddev = predictive_dist.stddev()
        P = norm.pdf(Y-3, loc=mean, scale=stddev) # calc pdf for samples based on predictive mean and stddev

        return np.reshape(P,(self.n_samples,1,1))
        
    def target_pdf(self, params): 
        return jax.scipy.stats.multivariate_normal.logpdf(x=params, mean=self.target_mean, cov=self.target_stddev).sum()  # TODO - compare performance with pdf vs log pdf
 
    def hmc(self):
        params_init = jnp.zeros((5,4))  # determines the sample shape..?
        start = time.time()
        chain = hmc.sample(self.key, params_init, self.target_pdf, n_steps=100, n_leapfrog_steps=100, step_size=0.1)
        end = time.time()

        print(f"HMC took {end - start} ms")
        # print(chain.shape)
        return chain

    
if __name__ == "__main__":
    gp = GaussianProcessSampling()
    gp.run()
    gp.hmc()