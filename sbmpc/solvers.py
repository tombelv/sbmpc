from sbmpc.model import BaseModel
from sbmpc.settings import Config

import jax.numpy as jnp
import jax

from functools import partial

from abc import ABC, abstractmethod


class BaseObjective(ABC):
    def __init__(self, robot_model=None):
        self.robot_model = robot_model

    @abstractmethod
    def running_cost(self, state, inputs, reference):
        pass

    def final_cost(self, state, reference):
        return 0.0


class SamplingBasedMPC:
    """
    Sampling-based MPC solver.
    """
    def __init__(self, model: BaseModel, objective: BaseObjective, config: Config):
        """
        Initializes the solver with the model, the objective, configurations and initial guess.
        Parameters
        ----------
        model: BaseModel
            The model propagated during rollouts.
        objective: BaseObjective
            Required to compute the cost function in the rollout.
        config_mpc: ConfigMPC
            Contains the MPC related parameters such as the time horizon, number of samples, etc.
        """

        self.model = model
        self.objective = objective

        # Sampling time for discrete time model
        self.dt = config.MPC["dt"]
        # Control horizon of the MPC (steps)
        self.horizon = config.MPC["horizon"]
        # Monte-carlo samples, that is the number of trajectories that are evaluated in parallel
        self.num_parallel_computations = config.MPC["num_parallel_computations"]
        
        # Covariance of the input action
        self.sigma_mppi = jnp.diag(config.MPC["std_dev_mppi"]**2)
        
        # Total number of inputs over time (stored in a 1d vector)
        self.num_control_variables = model.nu * self.horizon

        self.dtype_general = config.general["dtype"]
        self.device = config.general["device"]

        self.input_max_full_horizon = jnp.tile(model.input_max, self.horizon)
        self.input_min_full_horizon = jnp.tile(model.input_min, self.horizon)

        self.clip_input = jax.jit(self.clip_input, device=self.device)

        self.std_dev_horizon = jnp.tile(config.MPC["std_dev_mppi"], self.horizon)

        # Initialize the vector storing the current optimal input sequence
        if config.MPC["initial_guess"] is None:
            self.initial_guess = 0.0 * config.MPC["std_dev_mppi"]
            self.best_control_vars = jnp.zeros((self.num_control_variables,), dtype=self.dtype_general)
        else:
            self.initial_guess = config.MPC["initial_guess"]
            self.best_control_vars = jnp.tile(self.initial_guess, self.horizon)

        self.filter = config.MPC["filter"]
        if self.filter is not None:
            self.last_inputs_window = jnp.tile(self.initial_guess, self.filter.window_size//2)
        else:
            self.last_inputs_window = jnp.tile(self.initial_guess, 1)

        self.master_key = jax.random.PRNGKey(420)
        self.initial_random_parameters = jnp.zeros((self.num_parallel_computations, self.num_control_variables),
                                                   dtype=self.dtype_general)



        # if isinstance(model, Model):
        #     self.batch_state = self._batch_state
        # elif isinstance(model, ModelMjx):
        #     self.batch_state = self._batch_state_mjx
        # else:
        #     raise ValueError("""
        #                 Invalid model.
        #                 """)

        # self.vectorized_rollout = jax.vmap(self.compute_rollout, in_axes=(None, None, 0), out_axes=0)
        # self.jit_vectorized_rollout = jax.jit(self.vectorized_rollout, device=config_general.device)

        # Jit the controller function
        self.jit_compute_control_mppi = jax.jit(self.compute_control_mppi, device=self.device)

        # self.control_sens = jax.jit(jax.jacfwd(self.compute_control_mppi, argnums=0), device=config_general.device)

        self.running_cost = jax.jit(jax.vmap(self.objective.running_cost, in_axes=(0, 0, None), out_axes=0),
                                    device=self.device)

        self.final_cost = jax.jit(jax.vmap(self.objective.final_cost, in_axes=(0, None), out_axes=0),
                                  device=self.device)

    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def clip_input(self, control_variables):
        return jnp.clip(control_variables, self.input_min_full_horizon, self.input_max_full_horizon)

    # def compute_rollout(self, initial_state, reference, control_variables):
    #     """
    #     !!! As of now this only works for the classic model !!!
    #
    #     Calculate cost of a rollout of the dynamics given random control variables.
    #     Parameters
    #     ----------
    #     initial_state : jnp.array
    #         Initial state of the rollout.
    #     reference : jnp.array
    #         Desired state of the robot
    #     control_variables
    #         Vector of control inputs over the trajectory
    #     Returns
    #     -------
    #     cost : float
    #         cost of the rollout
    #     """
    #     def iterate_fun(idx, carry):
    #         sum_cost, curr_state, reference = carry
    #
    #         current_input = jax.lax.dynamic_slice_in_dim(control_variables, idx * self.model.nu, self.model.nu)
    #
    #         running_cost = self.objective.running_cost(curr_state, current_input, reference[idx, :])
    #
    #         # Integrate the dynamics
    #         state_next = self.model.integrate(curr_state, current_input, self.dt)
    #
    #         return sum_cost + running_cost, state_next, reference
    #
    #     carry = (jnp.float32(0.0), initial_state, reference)
    #     cost, state, reference = jax.lax.fori_loop(0, self.horizon, iterate_fun, carry)
    #
    #     cost = cost + self.objective.final_cost(state, reference[self.horizon, :])
    #
    #     return cost

    def _rollout(self, initial_state, reference, control_variables):

        cost = jnp.zeros(self.num_parallel_computations)

        curr_state = self._batch_state(initial_state)

        for idx in range(self.horizon):
            running_cost, curr_state = self._rollout_step(curr_state, reference, control_variables, idx)
            cost += running_cost

        cost += self.final_cost(curr_state, reference[self.horizon, :])

        return cost

    def _rollout_step(self, state, reference, control_variables, idx):
        current_input = jax.lax.dynamic_slice(control_variables,
                                              (0, idx * self.model.nu),
                                              (self.num_parallel_computations, self.model.nu))
        running_cost = self.running_cost(state, current_input, reference[idx, :])
        # Integrate the dynamics
        state_next = self.model.integrate_rollout(state, current_input, self.dt)
        return running_cost, state_next

    def compute_control_mppi(self, state, reference, best_control_vars, key):
        """
        This function computes the control parameters by applying MPPI.
        Parameters
        ----------
        state : jnp.array
            The current state of the robot for feedback
        reference : jnp.array
            The desired state of the robot
        best_control_vars : jnp.array
            The solution guess from the previous iterations
        key
         RNG key
        Returns
        -------
        optimal_action : jnp.array
            The optimal input trajectory shaped (num_control_variables, )
        """

        # Generate random parameters
        # The first control parameters is the old best one, so we add zero noise there
        additional_random_parameters = self.initial_random_parameters * 0.0

        # GAUSSIAN
        num_sample_gaussian_1 = self.num_parallel_computations - 1

        # Multivariate sampling with different standard deviation for the different inputs (slower implementation)
        # sampled_variation = self.moving_average.filter_batch(jax.random.multivariate_normal(key,
        #                                                    mean=jnp.zeros(self.model.nu),
        #                                                    cov=self.sigma_mppi,
        #                                                    shape=(num_sample_gaussian_1, self.horizon)).reshape(
        #     num_sample_gaussian_1, self.num_control_variables))

        sampled_variation = jax.random.normal(key=key, shape=(num_sample_gaussian_1, self.num_control_variables)) * self.std_dev_horizon

        additional_random_parameters = additional_random_parameters.at[1:self.num_parallel_computations].set(sampled_variation)

        if self.filter is not None:
            # Compute the candidate control sequences considering input constraints
            control_vars_all = self.filter.filter_batch(self.clip_input(best_control_vars + additional_random_parameters),
                                                            self.last_inputs_window,
                                                            jnp.array(()))
        else:
            control_vars_all = self.clip_input(best_control_vars + additional_random_parameters)

        additional_random_parameters_clipped = control_vars_all - best_control_vars

        # Do rollout
        costs = self._rollout(state, reference, control_vars_all)

        # Compute MPPI update
        cost, best_cost, worst_cost = self._sort_and_clip_costs(costs)
        exp_costs = self._exp_costs_invariant(costs, best_cost, worst_cost)
        # exp_costs = self._exp_costs_shifted(costs, best_cost)

        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom
        weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * additional_random_parameters_clipped.reshape(
            (self.num_parallel_computations, self.num_control_variables, 1))
        best_control_vars += jnp.sum(weighted_inputs, axis=0).reshape((self.num_control_variables,))

        return best_control_vars#, best_cost, costs

    def command(self, state, reference, shift_guess=True, num_steps=1):
        """
        This function computes the control action by applying MPPI.
        Parameters
        ----------
        state : jnp.array
            The current state of the robot for feedback
        reference
            The desired state of the robot or reference trajectory (dim: horizon x nx)
        shift_guess : bool (default = True)
            Determines if the resulting control action is stored in a shifted version of the control variables
        num_steps : int
            How many steps of optimization to make before returning the solution
        Returns
        -------
        optimal_action : jnp.array
            The optimal input trajectory shaped (num_control_variables, )
        """

        # If the reference is just a state, repeat it along the horizon
        if reference.ndim == 1:
            reference = jnp.tile(reference, (self.horizon+1, 1))

        best_control_vars = self.best_control_vars
        # maybe this loop should be jitted to actually be more efficient
        for i in range(num_steps):
            best_control_vars = self.jit_compute_control_mppi(state,
                                                                    reference,
                                                                    best_control_vars,
                                                                    self.master_key)
            # grad = self.control_sens(state, reference, best_control_vars, self.master_key)
            self._update_key()

        if shift_guess:
            self.last_inputs_window, self.best_control_vars = self._shift_guess(self.last_inputs_window, best_control_vars)
        else:
            self.last_inputs_window = self.last_inputs_window.at[-self.model.nu:].set(
                best_control_vars[:self.model.nu])
            self.best_control_vars = best_control_vars

        return best_control_vars

    def _sort_and_clip_costs(self, costs):
        # Saturate the cost in case of NaN or inf
        costs = jnp.where(jnp.isnan(costs), 1000000., costs)
        costs = jnp.where(jnp.isinf(costs), 1000000., costs)
        # Take the best found control parameters
        best_index = jnp.nanargmin(costs)
        worst_index = jnp.nanargmax(costs)
        best_cost = costs.take(best_index)
        worst_cost = costs.take(worst_index)

        return costs, best_cost, worst_cost

    def _exp_costs_shifted(self, costs, best_cost):

        lam = 1.
        exp_costs = jnp.exp(- lam * (costs - best_cost))

        return exp_costs

    def _exp_costs_invariant(self, costs, best_cost, worst_cost):
        """
        For a comparison see:
        G. Rizzi, J. J. Chung, A. Gawel, L. Ott, M. Tognon and R. Siegwart,
        "Robust Sampling-Based Control of Mobile Manipulators for Interaction With Articulated Objects,"
        in IEEE Transactions on Robotics, vol. 39, no. 3, pp. 1929-1946, June 2023, doi: 10.1109/TRO.2022.3233343.
        """

        h = 20.
        exp_costs = jnp.exp(- h * (costs - best_cost) / (worst_cost - best_cost))

        return exp_costs

    def _update_key(self):
        newkey, subkey = jax.random.split(self.master_key)
        self.master_key = newkey

    # def _batch_state_mjx(self, state):
    #     state_vec = jnp.concatenate([state.qpos, state.qvel])
    #     return jnp.tile(state_vec, (self.num_parallel_computations, 1))

    def _batch_state(self, state):
        state = jnp.tile(state, (self.num_parallel_computations, 1))
        return state

    @partial(jax.jit, static_argnums=(0,), device=jax.devices("cpu")[0])
    def _shift_guess(self, last_inputs_window, best_control_vars):
        last_inputs_window_new = jnp.roll(last_inputs_window, shift=-self.model.nu, axis=0)
        last_inputs_window_new = last_inputs_window_new.at[-self.model.nu:].set(
            best_control_vars[:self.model.nu])
        best_control_vars_shifted = jnp.roll(best_control_vars, shift=-self.model.nu, axis=0)
        best_control_vars_shifted = best_control_vars_shifted.at[-self.model.nu:].set(
            best_control_vars_shifted[-2 * self.model.nu:-self.model.nu])

        return last_inputs_window_new, best_control_vars_shifted

