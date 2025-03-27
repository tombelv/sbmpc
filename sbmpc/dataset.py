import jax.numpy as jnp
import numpy as np
import pandas as pd

import sbmpc.settings as settings
from sbmpc.solvers import Controller
from sbmpc.simulation import build_model_from_config, BgSimulator 
from examples.quadrotor import quadrotor_dynamics
from examples.quadrotor_obstacles import Objective
from sbmpc.obstacle_loader import ObstacleLoader

from sbmpc.sampler import MPPISampler
from sbmpc.solvers import RolloutGenerator
from sbmpc.gains import  MPPIGain

MASS = 0.027
GRAVITY = 9.81
INPUT_HOVER = jnp.array([MASS*GRAVITY, 0., 0., 0.], dtype=jnp.float32)

obsl = ObstacleLoader()

num_steps = 100
num_samples = 10
sim_iters = 500
horizon = 25
              
class DataSet():
    def __init__(self):
        pass

    def create(self, num_samples):  # num_steps = number of steps to simulate, num_samples - number of parallel computations to take from each step
        all_samples = None
        total_samples = num_steps*num_samples
        
        trajs = ["circle", "diagonal", "sine"]
        ratios = [0.4,0.4,0.2]
        obs_trajs = [obsl.get_obstacle_trajectory(sim_iters,traj)[:horizon+1] for traj in trajs] 
            
        samples = []
        for i in range(3):
            traj_samples = self.get_quadrotor_rollouts(num_samples, obs_trajs[i])
            np.random.shuffle(traj_samples)
            samples.append(traj_samples[:int(total_samples*ratios[i])])

        # all_samples = self.get_quadrotor_rollouts(num_samples)  # rollout with single trajectory
        all_samples = np.concatenate(samples) 
        np.random.shuffle(all_samples)
        # all_samples[:,-1] = np.abs(all_samples[:,-1])

        np.savetxt("/home/ubuntu/sbmpc/sbmpc/datasets/dataset_2.data", all_samples, fmt='%4.6f', delimiter=' ')     

        print(f"Created {all_samples.shape[0]} samples with {all_samples.shape[1]} dimensions") # ((num_samples * num_steps), 114)
  
    def get_quadrotor_rollouts(self, num_samples, traj): # simulation setup from quadrotor.py
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

        reference = jnp.concatenate((x_des, INPUT_HOVER))  
        reference = jnp.tile(reference, (horizon, 1))

        # traj = obsl.get_obstacle_trajectory(config.sim_iterations,"sine")[:horizon] 

        reference = jnp.concatenate([reference, traj],axis=1)

        bg_sim = self.build_solver_and_simulator(reference)

        rows = num_samples * num_steps

        initial_states_all = np.zeros((rows,13))  
        control_vars_all = np.zeros((rows,100))
        # control_vars_all = np.zeros((rows,25)) # dimensionality reduction
        constraint_violation_all = np.zeros((rows,1))

        for i in range(num_steps): # take first 10 rollouts for each state (currently 100 states) 
            initial_states, control_vars, constraint_violation = bg_sim.run()[0] # returns 2000 parallel rollouts for given state
            initial_states = np.array(initial_states)[:num_samples]

            # control_vars = control_vars[:,:,0][:num_samples] # dimensionality reduction
            # print(control_vars.shape)

            # control_vars = np.reshape(control_vars[:,:,0],(2000,25,1))
            # control_vars = control_vars[:,:,0]
            control_vars = np.reshape(control_vars,(control_vars.shape[0],-1))[:num_samples] # flatten control vars (25,4) -> (100,)

            constraint_violation = np.reshape(constraint_violation,(constraint_violation.shape[0], 1))[:num_samples]
            # print(control_vars.shape)

            idx = i*num_samples
            # all_samples[idx:idx+num_samples] = samples

            initial_states_all[idx:idx+num_samples] = initial_states
            control_vars_all[idx:idx+num_samples] = control_vars
            constraint_violation_all[idx:idx+num_samples] = constraint_violation 

        rollouts = np.float64(np.concatenate([initial_states_all, control_vars_all, constraint_violation_all], axis=1))
        # rollouts = np.float64(np.concatenate([initial_states_all,  constraint_violation_all], axis=1))[:,:5]

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
        rollout_generator = RolloutGenerator(solver_dynamics_model, quad_obj, config)
        gains = MPPIGain(config)
        sampler = MPPISampler(config)  # use MPPI sampler for generating rollouts
        controller = Controller(rollout_generator, sampler, gains)
        sim = BgSimulator(sim_state_init, sim_dynamics_model, rollout_generator, sampler, gains, controller, reference=reference, visualize=False, obstacles=True)
        return sim               


if __name__ == "__main__":
    dataset = DataSet()
    dataset.create(num_samples=num_samples)