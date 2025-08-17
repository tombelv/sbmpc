from abc import ABC, abstractmethod
import jax
import jax.experimental
import jax.numpy as jnp
from sbmpc.settings import Config
from sbmpc.sampler_models import GaussianProcessModel, BNNSampling

from functools import partial
import numpy as np

class Sampler(ABC):
    def __init__(self, config: Config) -> None:
        # Initialize the vector storing the current optimal input sequence
        self.horizon = config.MPC.horizon
        self.num_control_points = config.MPC.num_control_points
        self.model_nu = config.robot.nu
        self.lam = config.MPC.lambda_mpc
        self.std_dev = config.MPC.std_dev_mppi
        self.std_dev_horizon = jnp.tile(self.std_dev, self.num_control_points)
        self.dtype_general = config.general.dtype
        # Monte-carlo samples, that is the number of trajectories that are evaluated in parallel
        self.num_parallel_computations = config.MPC.num_parallel_computations
        if config.MPC.initial_guess is None:
            self.optimal_samples = jnp.zeros((self.horizon, self.model_nu), dtype=self.dtype_general)
        else:
            self.optimal_samples = jnp.tile(config.MPC.initial_guess, (self.horizon, 1))
        # scaffolding for storing all the control actions on the prediction horizon for each rollout
        self.zero_random_deviations = jnp.zeros((self.num_parallel_computations, self.num_control_points, self.model_nu), dtype=self.dtype_general)

        # this is initialized at the first interation
        self.additional_random_samples_clipped = None
        self.master_key = jax.random.PRNGKey(420) 

    @abstractmethod
    def sample_input_sequence(self,key) -> jnp.ndarray:
        pass

    #def compute_next_best(self,samples, costs) -> jnp.ndarray:
    #    pass

    @abstractmethod
    def update(self, initial_guess, samples, costs) -> jnp.ndarray:
        pass

    def _update_key(self):
        newkey, subkey = jax.random.split(self.master_key)
        self.master_key = newkey

    def write(self,num_rej):
        self.num_sample_rejections.append(num_rej)
        f = open("/home/ubuntu/sbmpc/experiments/sample_rejections.txt", "w")
        f.write(str(np.array(self.num_sample_rejections)))
        f.close()
        return num_rej
    

class GPSampler(Sampler):
    def __init__(self,config: Config) -> None:
        super().__init__(config)
        self.model = GaussianProcessSampling(self.num_parallel_computations) # initialise gp
        self.zero_random_deviations = jnp.zeros((self.num_parallel_computations, self.num_control_points, self.model_nu), dtype=self.dtype_general)
        self.num_sample_rejections = []

    @partial(jax.jit, static_argnums=(0,))
    def sample_input_sequence(self, key, state) -> jnp.ndarray: 
        samples_delta = self.zero_random_deviations
        sampled_variation_all = jax.random.normal(key=key, shape=(self.num_parallel_computations-1, self.num_control_points, self.model_nu)) * self.std_dev # sample from normal gaussian
        samples_delta = samples_delta.at[1:, :, :].set(sampled_variation_all)  
 
        return samples_delta

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, initial_guess, samples_delta, costs, rejections) -> jnp.ndarray:
        exp_costs = self._exp_costs_shifted(costs, jnp.min(costs))
        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom
        constraint_violation = self.model.get_P(samples_delta)
        weighted_inputs = (weights[:, jnp.newaxis, jnp.newaxis] + constraint_violation) * samples_delta
        # weighted_inputs = (weights[:, jnp.newaxis, jnp.newaxis] + constraint_violation) * samples_delta
        weighted_inputs = (np.matmul(weights[:, jnp.newaxis, jnp.newaxis],constraint_violation)) * samples_delta
               
        jax.experimental.io_callback(callback=self.write, num_rej=rejections, 
                                     result_shape_dtypes=jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32)) # record rejection rate
        return optimal_action
    
    def update(self, initial_guess, samples_delta, costs, rejections) -> jnp.ndarray:
        optimal_action = self.compute_action(initial_guess, samples_delta, costs, rejections)
        self._update_key()
        return optimal_action
    
    def _exp_costs_shifted(self, costs, best_cost) -> jnp.ndarray:
        return jnp.exp(- self.lam * (costs - best_cost))
    
    
class BNNSampler(Sampler):
    def __init__(self,config: Config) -> None:
        super().__init__(config)
        self.model = BNNSampling(self.num_parallel_computations) 
        self.zero_random_deviations = jnp.zeros((self.num_parallel_computations, self.num_control_points, self.model_nu), dtype=self.dtype_general)
        self.num_sample_rejections = []
        
    @partial(jax.jit, static_argnums=(0,))
    def sample_input_sequence(self, key, state) -> jnp.ndarray: 
        samples_delta = self.zero_random_deviations
        sampled_variation_all = jax.random.normal(key=key, shape=(self.num_parallel_computations-1, self.num_control_points, self.model_nu)) * self.std_dev # sample from normal gaussian
        samples_delta = samples_delta.at[1:, :, :].set(sampled_variation_all)  
 
        return samples_delta

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, initial_guess, samples_delta, costs, rejections) -> jnp.ndarray:
        exp_costs = self._exp_costs_shifted(costs, jnp.min(costs))
        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom
        constraint_violation = self.model.get_P(samples_delta)  
        # weighted_inputs = (weights[:, jnp.newaxis, jnp.newaxis] + constraint_violation) * samples_delta
        weighted_inputs = (np.matmul(weights[:, jnp.newaxis, jnp.newaxis],constraint_violation)) * samples_delta
        
        optimal_action = initial_guess + jnp.sum(weighted_inputs, axis=0) 
              
        jax.experimental.io_callback(callback=self.write, num_rej=rejections, 
                                     result_shape_dtypes=jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32)) # record rejection rate
        return optimal_action
    
    def update(self, initial_guess, samples_delta, costs, rejections) -> jnp.ndarray:
        optimal_action = self.compute_action(initial_guess, samples_delta, costs, rejections)
        self._update_key()
        return optimal_action
    
    def _exp_costs_shifted(self, costs, best_cost) -> jnp.ndarray:
        return jnp.exp(- self.lam * (costs - best_cost))
    

class MPPISampler(Sampler):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.f = None
        self.num_sample_rejections = []
        
    @partial(jax.jit, static_argnums=(0,))
    def sample_input_sequence(self, key, state) -> jnp.ndarray:  
        # Generate random samples
        samples_delta = self.zero_random_deviations
        # One sample is kept equal to the guess
        sampled_variation_all = jax.random.normal(key=key, shape=(self.num_parallel_computations-1, self.num_control_points, self.model_nu)) * self.std_dev
        samples_delta = samples_delta.at[1:, :, :].set(sampled_variation_all)

        return sampled_variation_all

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, initial_guess, samples_delta, costs, rejections) -> jnp.ndarray:
        exp_costs = self._exp_costs_shifted(costs, jnp.min(costs))
        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom
        
        jax.experimental.io_callback(callback=self.write, num_rej=rejections, 
                                     result_shape_dtypes=jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32)) # record rejection rate

        weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * samples_delta
        optimal_action = initial_guess + jnp.sum(weighted_inputs, axis=0)
        return optimal_action
    
    def update(self, initial_guess, samples_delta, costs, rejections) -> jnp.ndarray:
        optimal_action = self.compute_action(initial_guess, samples_delta, costs, rejections)
        self._update_key()
        return optimal_action

    def _exp_costs_shifted(self, costs, best_cost) -> jnp.ndarray:
        return jnp.exp(- self.lam * (costs - best_cost))
    