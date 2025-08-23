import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sbmpc.settings as settings
import time
import numpy as np
import traceback

from sbmpc import BaseObjective
from sbmpc.simulation import build_all
from sbmpc.geometry import skew, quat_product, quat2rotm, quat_inverse
from sbmpc.obstacle_loader import ObstacleLoader
from sbmpc.sampler import Sampler, MPPISampler, GPSampler, BNNSampler, SimpleMPPISampler
# from experiments import utils


os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
jax.config.update("jax_default_matmul_precision", "high")

SCENE_PATH = "examples/bitcraze_crazyflie_2/scene.xml"
RESULTS_PATH = "/home/ubuntu/sbmpc/sbmpc/experiments/ecai/results/moving/"

INPUT_MAX = jnp.array([1, 2.5, 2.5, 2])
INPUT_MIN = jnp.array([0, -2.5, -2.5, -2])

MASS = 0.027
GRAVITY = 9.81
INERTIA = jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5], dtype=jnp.float32)
INERTIA_MAT = jnp.diag(INERTIA)

SPATIAL_INTERTIA_MAT = jnp.diag(jnp.concatenate([MASS*jnp.ones(3, dtype=jnp.float32), INERTIA]))
SPATIAL_INTERTIA_MAT_INV = jnp.linalg.inv(SPATIAL_INTERTIA_MAT)

INPUT_HOVER = jnp.array([MASS*GRAVITY, 0., 0., 0.], dtype=jnp.float32)
obsl = ObstacleLoader()


# N_OBS = 3 # REMEMBER TO CHANGE IN SIMULATION.PY
# N_OBS = 5
# N_OBS = 10
N_OBS = 15


@jax.jit
def quadrotor_dynamics(state: jnp.array, inputs: jnp.array, params: jnp.array) -> jnp.array:

    quat = state[3:7]
    ang_vel = state[10:13]

    orientation_mat = quat2rotm(quat)
    ang_vel_quat = jnp.array([0., state[10], state[11], state[12]])

    total_force = jnp.array([0., 0., inputs[0]]) - MASS*GRAVITY*orientation_mat[2, :]  # transpose + 3rd col = 3rd row

    total_torque = 1e-3*inputs[1:4] - skew(ang_vel) @ INERTIA_MAT @ ang_vel  # multiplication by normalization factor

    acc = SPATIAL_INTERTIA_MAT_INV @ jnp.concatenate([total_force, total_torque])

    state_dot = jnp.concatenate([state[7:10],
                                 0.5 * quat_product(quat, ang_vel_quat),
                                 orientation_mat @ acc[:3],
                                 acc[3:6]])

    return state_dot


class Objective(BaseObjective):
    """ Cost function for the Quadrotor regulation task"""
    def __init__(self):
        super().__init__()
        # self.num_sample_rejections = 0

    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        pos_err = state[0:3] - state_ref[0:3]
        att_err = quat_product(quat_inverse(state[3:7]), state_ref[3:7])[1:4]
        vel_err = state[7:10] - state_ref[7:10]
        ang_vel_err = state[10:13] - state_ref[10:13]

        return pos_err, att_err, vel_err, ang_vel_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:13]
        state_ref = state_ref.at[7:10].set(-1*(state[0:3] - state_ref[0:3]))
        input_ref = reference[13:13+4]
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        return (5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err +
                (inputs-input_ref).transpose() @ jnp.diag(jnp.array([10, 10, 10, 100])) @ (inputs-input_ref))

    def final_cost(self, state, reference):
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        return (10 * pos_err.transpose() @ pos_err +
                1 * att_err.transpose() @ att_err +
                5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err) 

    def constraints(self, state, inputs, reference):
        r = obsl.radius + 0.1                                           
        pos = state[0:3]
        n_obs = len(reference[17:])//3
        obs_pos = jnp.reshape(reference[17:],(n_obs,3))
        dist_from_obs = jnp.array([jnp.sum(abs(pos - obs) - r) for obs in obs_pos])  # l1 dist 
         
        return dist_from_obs, dist_from_obs
    
    def constraints_not_jit(self, state, inputs, reference):
        r = obsl.radius + 0.1                                           
        pos = state[0:3]
        n_obs = len(reference[17:])//3
        obs_pos = jnp.reshape(reference[17:],(n_obs,3))

        dist_from_obs = jnp.array([(abs(pos - obs) - r) for obs in obs_pos]) 

        def too_close(dist): # penalise only if x,y and z are in range
            if all(x < 0 for x in dist):
                return 1
            else:
                return 0
            
        dist_from_obs = [too_close(dist) for dist in dist_from_obs]

        return dist_from_obs 
    
    def write(self,num_rej):
            self.num_sample_rejections.append(num_rej)
            f = open("/home/ubuntu/sbmpc/experiments/sample_rejections.txt", "w")
            f.write(str(np.array(self.num_sample_rejections)))
            f.close()
            return num_rej
    
class ConstraintFreeObjective(BaseObjective):
    """ Cost function for the Quadrotor regulation task"""
    def __init__(self):
        super().__init__()
        # self.num_sample_rejections = 0

    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        pos_err = state[0:3] - state_ref[0:3]
        att_err = quat_product(quat_inverse(state[3:7]), state_ref[3:7])[1:4]
        vel_err = state[7:10] - state_ref[7:10]
        ang_vel_err = state[10:13] - state_ref[10:13]

        return pos_err, att_err, vel_err, ang_vel_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:13]
        state_ref = state_ref.at[7:10].set(-1*(state[0:3] - state_ref[0:3]))
        input_ref = reference[13:13+4]
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        return (5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err +
                (inputs-input_ref).transpose() @ jnp.diag(jnp.array([10, 10, 10, 100])) @ (inputs-input_ref))

    def final_cost(self, state, reference):
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        return (10 * pos_err.transpose() @ pos_err +
                1 * att_err.transpose() @ att_err +
                5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err) 

    def constraints(self, state, inputs, reference):
        r = obsl.radius + 0.1                                           
        pos = state[0:3]
        n_obs = len(reference[17:])//3
        obs_pos = jnp.reshape(reference[17:],(n_obs,3))
        dist_from_obs = jnp.array([jnp.sum(abs(pos - obs) - r) for obs in obs_pos])  # l1 dist 
        neutral_constraints = jnp.array([1 for obs in obs_pos])
         
        return neutral_constraints, dist_from_obs
    
    def constraints_not_jit(self, state, inputs, reference):
        r = obsl.radius + 0.1                                           
        pos = state[0:3]
        n_obs = len(reference[17:])//3
        obs_pos = jnp.reshape(reference[17:],(n_obs,3))

        dist_from_obs = jnp.array([(abs(pos - obs) - r) for obs in obs_pos]) 

        def too_close(dist): # penalise only if x,y and z are in range
            if all(x < 0 for x in dist):
                return 1
            else:
                return 0
            
        dist_from_obs = [too_close(dist) for dist in dist_from_obs]

        return dist_from_obs 
    
    def write(self,num_rej):
            self.num_sample_rejections.append(num_rej)
            f = open("/home/ubuntu/sbmpc/experiments/sample_rejections.txt", "w")
            f.write(str(np.array(self.num_sample_rejections)))
            f.close()
            return num_rej


def avg_dist_from_obs(state_traj, obs_ref, n_obs):
        print(f"State shape = {state_traj.shape} and ref shape = {obs_ref.shape}")
        r = 0.05            
        n_iters = 500                     

        total_dist = 0
        for i in range(n_iters):
            curr_pos = state_traj[i]
            curr_obs_pos = obs_ref[i]
            for obs in curr_obs_pos:
                total_dist += jnp.sum(abs(curr_pos - obs) - r) 

        avg_dist = ((total_dist)/n_iters)/n_obs
        return avg_dist

def avg_dist_from_target(state_traj, target):
        n_iters = 100
        target_coords = target[:3]            

        total_dist = 0
        for i in range(n_iters):
            curr_pos = state_traj[i]
            total_dist += jnp.sum(abs(curr_pos - target_coords)) 

        avg_dist = ((total_dist)/n_iters)
        return avg_dist

def reached_target(avg_dist_from_target):
        if avg_dist_from_target < 0.1:
            return True
        return False

def num_collisions(state_traj, obs_ref):
        r = 0.05      
        n_iters = 500
       
        def too_close(dist): # penalise only if x,y and z are in range
            if all(x < 0 for x in dist):
                return 1
            else:
                return 0
            
        num_collisions = 0    
        for i in range(n_iters):
            curr_pos = state_traj[i]
            curr_obs_pos = jnp.reshape(obs_ref[i],(N_OBS,3))
            dist_from_obs = jnp.array([(abs(curr_pos - obs) - r) for obs in curr_obs_pos]) 
            num_collisions += np.sum([too_close(dist) for dist in dist_from_obs])
        
        return num_collisions

def write_results_to_csv(n, traj, mppi_data, mppi_base_data, bnn_data):
    metrics = {
        "rejection rate.csv": (mppi_data[2], mppi_base_data[2], bnn_data[2]),
        "control frequency.csv": (mppi_data[1], mppi_base_data[1], bnn_data[1]),
        "simulation runtime.csv": (mppi_data[0], mppi_base_data[0], bnn_data[0]),
        "average obstacle dist.csv": (mppi_data[3], mppi_base_data[3], bnn_data[3]),
        "num collisions.csv": (mppi_data[4], mppi_base_data[4], bnn_data[4]),
        "average target dist.csv": (mppi_data[5], mppi_base_data[5], bnn_data[5]),
        "reached target.csv": (mppi_data[6], mppi_base_data[6], bnn_data[6])
    }
    
    for filename, (mppi_val, mppi_base_val, bnn_val) in metrics.items():
        with open(RESULTS_PATH + filename, "a") as f:
            f.write(f"\n{n},{traj},{mppi_val},{mppi_base_val},{bnn_val}")

def get_simulation_results(sampler, samples):
        objective = Objective()
        sim = build_all(config, objective,
                        reference,
                        custom_dynamics_fn=quadrotor_dynamics,sampler=sampler)

        start = time.time()
        sim.simulate()   # start simulation
        end = time.time()
    
        duration = end - start 
        obstacle_dist = avg_dist_from_obs(sim.state_traj[:, 0:3], full_traj, N_OBS)
        n_collisions = num_collisions(sim.state_traj[:, 0:3], full_traj)
        
        with open("experiments/sample_rejections.txt", "r") as file:
            data = file.read()

        sample_rejections = np.fromstring(data.strip("[]"), sep=" ", dtype=int) 
        rejection_rate = (np.mean(sample_rejections)/N_OBS/samples)*100  # calculate rejection rate (in %)

        with open("experiments/computation_time.txt", "r") as file:
            data = file.read()
        computation_times = np.fromstring(data.strip("[]"), sep=" ", dtype=int)

        avg_comp_time = np.mean(1/(computation_times/1000))  # calculate control frequency

        avg_dist_from_target_ = avg_dist_from_target(sim.state_traj[:, 0:3], q_des)
        reached_target_ = reached_target(avg_dist_from_target_)

        return duration, avg_comp_time, rejection_rate, obstacle_dist, n_collisions, avg_dist_from_target_, reached_target_

if __name__ == "__main__":
    try:
        obsl.n_obstacles = N_OBS
        obsl.create_obstacles()
        obsl.load_obstacles()

        robot_config = settings.RobotConfig()

        robot_config.robot_scene_path = SCENE_PATH
        robot_config.nq = 7
        robot_config.nv = 6
        robot_config.nu = 4
        robot_config.input_min = INPUT_MIN
        robot_config.input_max = INPUT_MAX
        robot_config.q_init = jnp.array([0., 0., 0., 1., 0., 0., 0.], dtype=jnp.float32)  # hovering position
        
        config = settings.Config(robot_config)

        config.general.visualize = True
        config.MPC.dt = 0.02
        config.MPC.horizon = 25
        config.MPC.std_dev_mppi = 0.2*jnp.array([0.1, 0.1, 0.1, 0.05])
        config.MPC.initial_guess = INPUT_HOVER
        config.MPC.lambda_mpc = 50.0
        config.MPC.smoothing = "Spline"
        config.MPC.num_control_points = 5
        config.MPC.gains = False

        config.solver_dynamics = settings.DynamicsModel.CUSTOM
        config.sim_dynamics = settings.DynamicsModel.MJX

        q_des = jnp.array(
        [0.5, 0.5, 0.5, 1., 0., 0., 0.], dtype=jnp.float32)  # hovering position
        x_des = jnp.concatenate([q_des, jnp.zeros(robot_config.nv, dtype=jnp.float32)], axis=0)

        horizon = config.MPC.horizon+1
        full_traj = obsl.get_obstacle_trajectory(config.sim_iterations,"circle")
        traj = full_traj[:horizon]  

        reference = jnp.concatenate((x_des, INPUT_HOVER))  
        reference = jnp.tile(reference, (horizon, 1))
        reference = jnp.concatenate([reference, traj],axis=1)

        # sample_sizes = [100, 250, 500, 750, 1000, 1250, 1500]
        sample_sizes = [500,1000]

        for n in sample_sizes: # TODO - set n_obstacles to 3 on simiulation line 72 and on line 143 here!!
            config.MPC.num_parallel_computations = n # set sample size

            mppi = MPPISampler(config) 
            duration_mppi, avg_comp_time_mppi, rej_rate_mppi, obs_dist_mppi, num_collisions_mppi, avg_target_dist_mppi, reached_target_mppi = get_simulation_results(mppi,n)
          
            mppi_base = SimpleMPPISampler(config)
            duration_base, avg_comp_time_base,  rej_rate_base, obs_dist_base, num_collisions_base, avg_target_dist_base, reached_target_base = get_simulation_results(mppi_base, n)

            bnn = BNNSampler(config)
            duration_bnn, avg_comp_time_bnn, rej_rate_bnn, obs_dist_bnn, num_collisions_bnn, avg_target_dist_bnn, reached_target_bnn  = get_simulation_results(bnn, n)

            mppi_results = (duration_mppi, avg_comp_time_mppi, rej_rate_mppi, obs_dist_mppi, num_collisions_mppi, avg_target_dist_mppi, reached_target_mppi)
            mppi_base_results = (duration_base, avg_comp_time_base, rej_rate_base, obs_dist_base, num_collisions_base, avg_target_dist_base, reached_target_base)
            bnn_results = (duration_bnn, avg_comp_time_bnn, rej_rate_bnn, obs_dist_bnn, num_collisions_bnn,avg_target_dist_bnn, reached_target_bnn)
            
            write_results_to_csv(N_OBS, n, mppi_results, mppi_base_results, bnn_results) 


        obsl.reset_xmls()       

    except Exception as e:
        print(f"Experiments terminated prematurely. Cause: {e}")
        traceback.print_exc()
        obsl.reset_xmls()
 