
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from sbmpc.settings import Config

from functools import partial

ZERO_RANDOM_DEVIATIONS = None
NUM_PARALLEL_COMPUTATIONS = None
NUM_CONTROL_POINTS = None
MODEL_NU = None
STD_DEV = None
OPTIMAL_SAMPLES = None

class Sampler(ABC):

    def __init__(self, config: Config) -> None:
        # Initialize the vector storing the current optimal input sequence
        global ZERO_RANDOM_DEVIATIONS, NUM_PARALLEL_COMPUTATIONS, NUM_CONTROL_POINTS, MODEL_NU, STD_DEV, OPTIMAL_SAMPLES
        self.horizon = config.MPC.horizon
        self.num_control_points = config.MPC.num_control_points
        NUM_CONTROL_POINTS = self.num_control_points
        self.model_nu = config.robot.nu
        MODEL_NU = self.model_nu
        self.lam = config.MPC.lambda_mpc
        self.std_dev = config.MPC.std_dev_mppi
        STD_DEV = self.std_dev
        self.std_dev_horizon = jnp.tile(self.std_dev, self.num_control_points)
        self.dtype_general = config.general.dtype
        # Monte-carlo samples, that is the number of trajectories that are evaluated in parallel
        self.num_parallel_computations = config.MPC.num_parallel_computations
        if config.MPC.initial_guess is None:
            self.optimal_samples = jnp.zeros((self.horizon, self.model_nu), dtype=self.dtype_general)
        else:
            self.optimal_samples = jnp.tile(config.MPC.initial_guess, (self.horizon, 1))
        OPTIMAL_SAMPLES = self.optimal_samples
        # scaffolding for storing all the control actions on the prediction horizon for each rollout
        ZERO_RANDOM_DEVIATIONS = jnp.zeros((self.num_parallel_computations, self.num_control_points, self.model_nu), dtype=self.dtype_general)
        NUM_PARALLEL_COMPUTATIONS = self.num_parallel_computations
        # this is initialized at the first interation
        self.additional_random_samples_clipped = None

        self.master_key = jax.random.PRNGKey(420)

        

    @staticmethod
    def sample_input_sequence(key) -> jnp.ndarray:
        pass

    #def compute_next_best(self,samples, costs) -> jnp.ndarray:
    #    pass

    @abstractmethod
    def update(self, initial_guess, samples, costs) -> jnp.ndarray:
        pass

    def _update_key(self):
        newkey, subkey = jax.random.split(self.master_key)
        self.master_key = newkey


class CEMSampler(Sampler):

    def __init__(self,config: Config) -> None:
        super().__init__(config)

    def sample_input_sequence(self, key) -> jnp.ndarray:
        # Return zero or your logic
        return jnp.zeros(
            (self.num_parallel_computations - 1, self.num_control_points, self.model_nu)
        )

    def update(self, initial_guess, samples, costs)-> jnp.ndarray:
        return initial_guess

class MPPISampler(Sampler):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
    @staticmethod
    @partial(jax.jit)
    def sample_input_sequence( key) -> jnp.ndarray:
        # Generate random samples
        samples_delta = jnp.copy(ZERO_RANDOM_DEVIATIONS)
        # One sample is kept equal to the guess
        sampled_variation_all = jax.random.normal(key=key, shape=(NUM_PARALLEL_COMPUTATIONS-1, NUM_CONTROL_POINTS, MODEL_NU)) * STD_DEV
        samples_delta = samples_delta.at[1:, :, :].set(sampled_variation_all)
        return sampled_variation_all

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, initial_guess, samples_delta, costs) -> jnp.ndarray:
        exp_costs = self._exp_costs_shifted(costs, jnp.min(costs))
        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom
        weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * samples_delta
        optimal_action = initial_guess + jnp.sum(weighted_inputs, axis=0)
        return optimal_action
    
    def update(self, initial_guess, samples_delta, costs) -> jnp.ndarray:
        optimal_action = self.compute_action(initial_guess, samples_delta, costs)
        self._update_key()
        return optimal_action

    def _exp_costs_shifted(self, costs, best_cost) -> jnp.ndarray:
        return jnp.exp(- self.lam * (costs - best_cost))
    

        

  


