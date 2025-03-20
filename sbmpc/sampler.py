
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from sbmpc.settings import Config
from sbmpc.gp import GaussianProcessSampling

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
    def update(self, parameters, costs, state=None) -> jnp.ndarray:
        pass


class CEMSampler(SBS):
    def __init__(self,config: Config, model_nu: int) -> None:
        super().__init__(config, model_nu)
        self.gp = GaussianProcessSampling() # initialise gp
        self.key = jax.random.PRNGKey(420)

    def sample_input_sequence(self, key) -> jnp.ndarray:
        additional_random_parameters = self.zero_random_deviations  # random sample
        sampled_variation_all = jax.random.normal(key=key, shape=(self.num_parallel_computations-1, self.num_control_points, self.model_nu)) * self.std_dev # (1999, 5, 4)
        additional_random_parameters = additional_random_parameters.at[1:, :, :].set(  # skipping the first element of the first dimension
            sampled_variation_all)
        return sampled_variation_all

    def update(self,parameters, costs, state)-> jnp.ndarray: # keep update step the same for now
        x,y,z = parameters.shape
        parameters = jnp.reshape(parameters, (x,y*z))
        state = jnp.tile(state,(x,1))
        flat_params = jnp.concatenate([state,parameters], axis=1)  # reshape parameters into shape that the model is expecting for predicction- (25,4) -> (113)
        
        self.gp.get_target_dist(flat_params) # skew params - weight with GP prob P
        samples = self.gp.hmc_sampling() # sample from skewed distribution
        optimal_control_vars = samples[-1,:,:] # get the end of the chain only - should be the best..
        
        self.best_control_vars = optimal_control_vars
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
        sampled_variation_all = jax.random.normal(key=key, shape=(self.num_parallel_computations-1, self.num_control_points, self.model_nu)) * self.std_dev # (1999, 5, 4)
        additional_random_parameters = additional_random_parameters.at[1:, :, :].set(  # skipping the first element of the first dimension
            sampled_variation_all)
        return sampled_variation_all

    def update(self,parameters, costs, state) -> jnp.ndarray: 
        # print(f"Parameters shape = {parameters.shape}") # (199,25,4)
        exp_costs = self._exp_costs_shifted(costs, jnp.min(costs)) # exponent of shifted costs
        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom  # average the costs
        weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * parameters # weight paraemers by their costs 
        optimal_action = self.best_control_vars + jnp.sum(weighted_inputs, axis=0) # update current best control vars
        #self.best_control_vars = optimal_action
        print(f"Optimal action shape = {optimal_action.shape}")
        return optimal_action
    
    
    def _exp_costs_shifted(self, costs, best_cost) -> jnp.ndarray:
        return jnp.exp(- self.lam * (costs - best_cost))
    

        

  


