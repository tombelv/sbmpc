
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from sbmpc.settings import Config

# TODO understand the role of initial_guess and intial_random_parameters and how to initialize best_control_vars at the beggining

class SBS(ABC):

    def __init__(self, config: Config, model_nu : int) -> None:
        # Initialize the vector storing the current optimal input sequence
        self.horizon = config.MPC.horizon
        self.num_control_points = config.MPC.num_control_points
        self.model_nu = model_nu
        self.lam = config.MPC.lambda_mpc
        self.std_dev = config.MPC.std_dev_mppi
        self.std_dev_horizon = jnp.tile(self.std_dev, self.num_control_points)
        self.dtype_general = config.general.dtype
        # Monte-carlo samples, that is the number of trajectories that are evaluated in parallel
        self.num_parallel_computations = config.MPC.num_parallel_computations
        if config.MPC.initial_guess is None:
            self.initial_guess = 0.0 * self.std_dev
            self.best_control_vars = jnp.zeros((self.horizon,self.model_nu), dtype=self.dtype_general)
        else:
            self.initial_guess = config.MPC.initial_guess
            self.best_control_vars = jnp.tile(self.initial_guess, (self.horizon, 1))
        # scaffolding for storing all the control actions on the prediction horizon for each rollout
        self.zero_random_deviations = jnp.zeros((self.num_parallel_computations, self.num_control_points, self.model_nu), dtype=self.dtype_general)

        # this is initialized at the first interation
        self.additional_random_parameters_clipped=None
        

    @abstractmethod
    def sample_input_sequence(self,key) -> jnp.ndarray:
        pass

    #def compute_next_best(self,parameters, costs) -> jnp.ndarray:
    #    pass

    @abstractmethod
    def update(self, parameters, costs) -> jnp.ndarray:
        pass


class CEMSampler(SBS):

    def __init__(self,config: Config, model_nu: int) -> None:
        super().__init__(config, model_nu)

    def sample_input_sequence(self, key) -> jnp.ndarray:
        # Return zero or your logic
        return jnp.zeros(
            (self.num_parallel_computations - 1, self.num_control_points, self.model_nu)
        )

    def update(self,parameters, costs)-> jnp.ndarray:
        return self.best_control_vars

class MPPISampler(SBS):
    # TODO initialize control point and parallel computationss
    def __init__(self, config: Config, model_nu: int) -> None:
        super().__init__(config, model_nu)
        
       
    def sample_input_sequence(self, key) -> jnp.ndarray:
        # Generate random parameters
        # The first control parameters is the old best one, so we add zero noise there
        additional_random_parameters = self.zero_random_deviations
        # One sample is kept equal to the guess
        sampled_variation_all = jax.random.normal(key=key, shape=(self.num_parallel_computations-1, self.num_control_points, self.model_nu)) * self.std_dev
        additional_random_parameters = additional_random_parameters.at[1:, :, :].set(
            sampled_variation_all)
        return sampled_variation_all

    def update(self, parameters, costs) -> jnp.ndarray:
        exp_costs = self._exp_costs_shifted(costs, jnp.min(costs))
        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom
        weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * parameters
        optimal_action = self.best_control_vars + jnp.sum(weighted_inputs, axis=0)
        #self.best_control_vars = optimal_action
        return optimal_action
    
    def _exp_costs_shifted(self, costs, best_cost) -> jnp.ndarray:
        return jnp.exp(- self.lam * (costs - best_cost))
    

        

  


