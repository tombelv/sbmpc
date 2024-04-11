from sbmpc.model import Model
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral

import jax.numpy as jnp
import jax


class SbMPC:
    def __init__(self, model: Model, cost_fn, config_mpc: ConfigMPC, config_general: ConfigGeneral):
        self.model = model
        self.cost_fn = cost_fn

        self.dt = config_mpc.dt
        self.horizon = config_mpc.horizon
        self.num_parallel_computations = config_mpc.num_parallel_computations
        self.sigma_mppi = config_mpc.sigma_mppi
        
        self.dtype_general = config_general.dtype_general

        self.num_control_variables = model.nu * self.horizon

        # Initialize the vector storing the current optimal input sequence
        self.best_control_parameters = jnp.zeros((self.num_control_variables,), dtype=self.dtype_general)

        self.master_key = jax.random.PRNGKey(420)
        self.initial_random_parameters = jnp.zeros((self.num_parallel_computations, self.num_control_variables),
                                                   dtype=self.dtype_general)

        self.vectorized_rollout = jax.vmap(self.compute_rollout, in_axes=(None, None, 0), out_axes=0)
        self.jit_vectorized_rollout = jax.jit(self.vectorized_rollout, device=config_general.device)

        # the first call of jax is very slow, hence we should do this since the beginning
        # creating a fake initial state, reference and contact sequence
        initial_state = jnp.zeros((self.model.nx,), dtype=self.dtype_general)
        initial_reference = jnp.zeros((self.model.nx,), dtype=self.dtype_general)

        control_parameters_vec = jax.random.uniform(self.master_key,
                                                    (self.num_control_variables * self.num_parallel_computations,),
                                                    minval=-100., maxval=100.)
        self.jit_vectorized_rollout(initial_state,
                                    initial_reference,
                                    control_parameters_vec.reshape(self.num_parallel_computations,
                                                                   self.num_control_variables))

        # Jit the controller function
        self.jit_compute_control_mppi = jax.jit(self.compute_control_mppi, device=config_general.device)

    def compute_rollout(self, initial_state, reference, control_variables):
        """Calculate cost of a rollout of the dynamics given random control variables.
        Args:
            initial_state (np.array): actual state of the robot
            reference (np.array): desired state of the robot
            control_variables (np.array): parameters for the controllers
        Returns:
            (float): cost of the rollout
        """

        state = initial_state
        cost = jnp.float32(0.0)

        def iterate_fun(iter, carry):
            cost, state, reference = carry

            current_input = jax.lax.dynamic_slice_in_dim(control_variables, iter*self.model.nu, self.model.nu)

            running_cost = self.cost_fn(state, reference[:self.model.nx], current_input)

            # Integrate the dynamics
            state_next = self.model.integrate(state, current_input, self.dt)

            return cost + running_cost, state_next, reference

        carry = (cost, state, reference)
        cost, state, reference = jax.lax.fori_loop(0, self.horizon, iterate_fun, carry)

        return cost

    def compute_control_mppi(self, state, reference, best_control_parameters, key):
        """
        This function computes the control parameters by applying MPPI.
        """
        # Generate random parameters
        # The first control parameters is the old best one, so we add zero noise there
        additional_random_parameters = self.initial_random_parameters * 0.0

        # GAUSSIAN
        num_sample_gaussian_1 = self.num_parallel_computations - 1
        additional_random_parameters = additional_random_parameters.at[1:self.num_parallel_computations].set(
            self.sigma_mppi * jax.random.normal(key=key, shape=(num_sample_gaussian_1, self.num_control_variables)))

        control_parameters_vec = best_control_parameters + additional_random_parameters

        # Do rollout
        costs = self.jit_vectorized_rollout(state, reference, control_parameters_vec)

        # Saturate the cost in case of NaN or inf
        costs = jnp.where(jnp.isnan(costs), 1000000., costs)
        costs = jnp.where(jnp.isinf(costs), 1000000., costs)

        # Take the best found control parameters
        best_index = jnp.nanargmin(costs)
        best_cost = costs.take(best_index)

        # # Compute MPPI update
        beta = best_cost
        temperature = 1.
        exp_costs = jnp.exp((-1. / temperature) * (costs - beta))
        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom
        weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * additional_random_parameters.reshape(
            (self.num_parallel_computations, self.num_control_variables, 1))
        best_control_parameters += jnp.sum(weighted_inputs, axis=0).reshape((self.num_control_variables,))

        return best_control_parameters, best_cost, costs

    def compute_control_action(self, state, reference):
        best_control_parameters, best_cost, costs = self.jit_compute_control_mppi(state,
                                                                                  reference,
                                                                                  self.best_control_parameters,
                                                                                  self.master_key)

        self.best_control_parameters = best_control_parameters

        return best_control_parameters



