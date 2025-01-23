import matplotlib as mpl  # keep as first import to avoid segmentation fault
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import gpjax as gpx
import sbmpc.settings as settings
import numpy as np
import jax.random as jr

from sbmpc.solvers import SamplingBasedMPC
from sbmpc.simulation import build_model_from_config, BG_Simulator
from examples.quadrotor import quadrotor_dynamics
from examples.quadrotor_obstacles import Objective
from sbmpc.obstacle_loader import ObstacleLoader

MASS = 0.027
GRAVITY = 9.81
INPUT_HOVER = jnp.array([MASS*GRAVITY, 0., 0., 0.], dtype=jnp.float32)
obsl = ObstacleLoader()
num_steps = 100

class DataSet():
    def __init__(self):
        pass

    def create(self, num_samples, make_new=True):  # num_steps = number of steps to simulate, num_samples - number of parallel computations to take from each step
        if make_new:
            samples = self.get_quadrotor_rollouts(num_samples) # ((num_samples * num_steps), 114)
            np.random.shuffle(samples)
            n = int(num_samples*num_steps*0.7)

            training_samples = samples[:n] # split into training and test sets - 70/30
            test_samples = samples[n:] 
            
            print(f"Training samples = {training_samples.shape} and test samples = {test_samples.shape}")
            # print(f"Inputs shape = {test_samples[:,:113].shape}, Outputs shape = {test_samples[:,113:].shape}")

            training_set = gpx.Dataset(X=training_samples[:,:113], y=training_samples[:,113:])  # split into inputs and outputs
            test_set = gpx.Dataset(X=test_samples[:,:113], y=test_samples[:,113:]) 

            # breakpoint()
            np.savetxt('/home/ubuntu/sbmpc/sbmpc/training.txt', training_samples, fmt='%4.6f', delimiter=' ') 
            np.savetxt('/home/ubuntu/sbmpc/sbmpc/test.txt', test_samples, fmt='%4.6f', delimiter=' ') 

            return training_set, test_set
    
        else: 
            # may not want to rebuild dataset each time, retrieve from file
            pass
        
    def get_quadrotor_rollouts(self, num_samples): 
        robot_config = settings.RobotConfig()
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
        traj = obsl.get_obstacle_trajectory(config.sim_iterations,"circle")[:horizon] 

        reference = jnp.concatenate((x_des, INPUT_HOVER))  
        reference = jnp.tile(reference, (horizon, 1))
        reference = jnp.concatenate([reference, traj],axis=1)

        bg_sim = self.build_solver_and_simulator(reference)

        # num_steps = 10
        rows = num_samples*num_steps
        initial_states_all = np.zeros((rows,13))  
        control_vars_all = np.zeros((rows,100))
        constraint_violation_all = np.zeros((rows,1))

        for i in range(num_steps): # take first 100 rollouts for each state (currently 100 states) TODO - sort by constraint violation
            initial_states, control_vars, constraint_violation = bg_sim.run()[0] # returns 2000 parallel rollouts for given state
            initial_states = np.array(initial_states)[:num_samples]
            control_vars = np.reshape(control_vars,(control_vars.shape[0],-1))[:num_samples]  # flatten last two dimensions
            constraint_violation = np.reshape(constraint_violation,(constraint_violation.shape[0], 1))[:num_samples]

            idx = i*num_samples
            initial_states_all[idx:idx+num_samples] = initial_states
            control_vars_all[idx:idx+num_samples] = control_vars
            constraint_violation_all[idx:idx+num_samples] = constraint_violation 

        rollouts = np.float64(np.concatenate([initial_states_all, control_vars_all, constraint_violation_all], axis=1))
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
        sim = BG_Simulator(sim_state_init, sim_dynamics_model, solver, reference, num_iter=100, visualizer = None, obstacles = True)

        return sim
                

class GaussianProcessSampling(): # https://docs.jaxgaussianprocesses.com/_examples/regression/#constructing-the-posterior
    def __init__(self):
        key, subkey = jr.split(jax.random.key(456))
        x = jr.uniform(key=jax.random.key(456), minval=-3.0, maxval=3.0, shape=(100,)).reshape(-1, 1)
        f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
        signal = f(x)
        y = signal + jr.normal(subkey, shape=signal.shape) * 0.3
        self.test  = gpx.Dataset(X=np.float64(x), y=np.float64(y))

        ds = DataSet()
        self.training_set, self.test_set = ds.create(num_samples=10)

        mean = gpx.mean_functions.Zero()  
        kernel = gpx.kernels.RBF()
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=100)

        self.prior = gpx.gps.Prior(mean_function = mean, kernel = kernel) 
        self.posterior = self.prior * likelihood
        self.opt_posterior = self.posterior           
    
    
    def optimise(self): # optimising kernel lengthscale, kernel variance and the observation noise variance
        print(-gpx.objectives.conjugate_mll(self.opt_posterior, self.training_set))

        self.opt_posterior, history = gpx.fit_scipy(
        model=self.posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),   # using marginal log likelihood as objective
        train_data=self.training_set,
        max_iters=500)

        print(history)
        print(f"*** Final error -> {-gpx.objectives.conjugate_mll(self.opt_posterior, self.training_set)} ***")
        # print(-gpx.objectives.conjugate_mll(self.opt_posterior, self.training_set))

    def predict(self, visualise=True):
        latent_dist = self.opt_posterior.predict(self.test_set.X, train_data=self.training_set)  # TODO - test
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        # predictive_mean = predictive_dist.mean()
        # predictive_std = predictive_dist.stddev()

        # print(f"*** Finished predicting -> {predictive_dist} ***")
     
    # def sample(self, visualise=True): # TODO - modify to sample from predictive distribution
    #     key = jax.random.key(456)
    #     prior_dist = self.prior.predict(self.dataset.X)
    #     # prior_dist = self.prior.predict(self.xtest)
    #     prior_mean = prior_dist.mean()
    #     prior_std = prior_dist.variance()
    #     samples = prior_dist.sample(seed=key, sample_shape=(20,))

    def run(self):
        # self.visualise_dataset()
        # self.sample(visualise=False)
        self.optimise()
        # self.predict(visualise=True)

if __name__ == "__main__":
    gp = GaussianProcessSampling()
    gp.run()