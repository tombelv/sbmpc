from sbmpc.new_model import BaseModel
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral

import jax.numpy as jnp
import jax


class SbMPC:
    def __init__(self, model: BaseModel, cost_fn, config_mpc: ConfigMPC, config_general: ConfigGeneral):
        self.model = model
        self.cost_fn = cost_fn

        self.dt = config_mpc.dt
        self.horizon = config_mpc.horizon
        self.num_parallel_computations = config_mpc.num_parallel_computations
        self.sigma_mppi = config_mpc.sigma_mppi

        self.dtype_general = config_general.dtype_general

        self.num_control_variables = model.nu * self.horizon

        self.input_max_full_horizon = jnp.repeat(model.input_max, self.horizon)
        self.input_min_full_horizon = jnp.repeat(model.input_min, self.horizon)

        clip_input_vectorized = jax.vmap(self.clip_input, in_axes=0, out_axes=0)
        self.clip_input = jax.jit(clip_input_vectorized, device=config_general.device)

        # Initialize the vector storing the current optimal input sequence
        self.best_control_vars = jnp.zeros((self.num_control_variables,), dtype=self.dtype_general)

        self.master_key = jax.random.PRNGKey(420)
        self.initial_random_parameters = jnp.zeros((self.num_parallel_computations, self.num_control_variables),
                                                   dtype=self.dtype_general)

        self.vectorized_rollout = jax.vmap(self.compute_rollout, in_axes=(None, None, 0), out_axes=0)
        self.jit_vectorized_rollout = jax.jit(self.vectorized_rollout, device=config_general.device)

        # Jit the controller function
        self.jit_compute_control_mppi = jax.jit(self.compute_control_mppi, device=config_general.device)

        # the first call of jax is very slow, hence we should do this since the beginning
        # creating a fake initial state, reference and contact sequence
        initial_state = jnp.zeros((self.model.nx,), dtype=self.dtype_general)
        initial_reference = jnp.zeros((self.model.nx,), dtype=self.dtype_general)

        self.jit_compute_control_mppi(initial_state, initial_reference, self.best_control_vars, self.master_key)

    def clip_input(self, control_variables):
        return jnp.clip(control_variables, self.input_min_full_horizon, self.input_max_full_horizon)

    def compute_rollout(self, initial_state, reference, control_variables):
        """Calculate cost of a rollout of the dynamics given random control variables.
        Args:
            initial_state (np.array): actual state of the robot
            reference (np.array): desired state of the robot
            control_variables (np.array): parameters for the controllers
        Returns:
            (float): cost of the rollout
        """
        def iterate_fun(idx, carry):
            sum_cost, curr_state, reference = carry

            current_input = jax.lax.dynamic_slice_in_dim(control_variables, idx * self.model.nu, self.model.nu)

            running_cost = self.cost_fn(curr_state, reference[:self.model.nx], current_input)

            # Integrate the dynamics
            state_next = self.model.integrate(curr_state, current_input, self.dt)

            return sum_cost + running_cost, state_next, reference

        carry = (jnp.float32(0.0), initial_state, reference)
        cost, state, reference = jax.lax.fori_loop(0, self.horizon, iterate_fun, carry)

        return cost

    def compute_control_mppi(self, state, reference, best_control_vars, key):
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

        control_vars_all = self.clip_input(best_control_vars + additional_random_parameters)

        additional_random_parameters_clipped = control_vars_all - best_control_vars

        # Do rollout
        costs = self.jit_vectorized_rollout(state, reference, control_vars_all)

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
        weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * additional_random_parameters_clipped.reshape(
            (self.num_parallel_computations, self.num_control_variables, 1))
        best_control_vars += jnp.sum(weighted_inputs, axis=0).reshape((self.num_control_variables,))

        return best_control_vars, best_cost, costs

    def compute_control_action(self, state, reference):
        best_control_vars, best_cost, costs = self.jit_compute_control_mppi(state,
                                                                            reference,
                                                                            self.best_control_vars,
                                                                            self.master_key)

        self.best_control_vars = best_control_vars

        return best_control_vars
