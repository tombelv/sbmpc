from sbmpc.model import BaseModel
from sbmpc.settings import Config

import jax.numpy as jnp
import jax

from functools import partial
from abc import ABC, abstractmethod

from sbmpc.filter import cubic_spline



class BaseObjective(ABC):
    def __init__(self, robot_model=None):
        self.robot_model = robot_model

    @abstractmethod
    def running_cost(self, state, inputs, reference):
        pass

    def final_cost(self, state, reference):
        return 0.0

    def cost_and_constraints(self, state, inputs, reference):
        return self.running_cost(state, inputs, reference) + jnp.sum(self.make_barrier(self.constraints(state, inputs, reference)))

    def final_cost_and_constraints(self, state, reference):
        return self.final_cost(state, reference) + jnp.sum(self.make_barrier(self.terminal_constraints(state, reference)))

    def make_barrier(self, constraint_array):
        constraint_array = jnp.where(constraint_array > 0, 1e3, 0.0)
        return constraint_array

    def constraints(self, state, inputs, reference):
        return 0.0

    def terminal_constraints(self, state, reference):
        return 0.0


@jax.jit
def _shift_guess(best_control_vars):
        best_control_vars_shifted = jnp.roll(best_control_vars, shift=-1, axis=0)
        best_control_vars_shifted = best_control_vars_shifted.at[-1, :].set(
            best_control_vars_shifted[-2:-1, :].reshape(-1))

        return best_control_vars_shifted


class SamplingBasedMPC():
    """
    Sampling-based MPC solver.
    
    This solver is static (apart from the master_key)
    
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

        self.config = config

        # Sampling time for discrete time model
        self.dt = config.MPC.dt
        # Control horizon of the MPC (steps)
        self.horizon = config.MPC.horizon
        # Monte-carlo samples, that is the number of trajectories that are evaluated in parallel
        self.num_parallel_computations = config.MPC.num_parallel_computations

        self.lam = config.MPC.lambda_mpc

        self.compute_gains = config.MPC.gains
        
        # Covariance of the input action
        # self.sigma_mppi = jnp.diag(config.MPC.std_dev_mppi**2)
        
        # Total number of inputs over time (stored in a 1d vector)
        self.num_control_variables = model.nu * self.horizon
        self.num_control_points = config.MPC.num_control_points
        self.control_points_sparsity = self.horizon // self.num_control_points

        self.dtype_general = config.general.dtype
        self.device = config.general.device

        self.input_max_full_horizon = jnp.tile(model.input_max, (self.horizon, 1))
        self.input_min_full_horizon = jnp.tile(model.input_min, (self.horizon, 1))

        self.clip_input = jax.jit(self.clip_input, device=self.device)

        self.std_dev = config.MPC.std_dev_mppi
        self.std_dev_horizon = jnp.tile(self.std_dev, self.num_control_points)

        # Initialize the vector storing the current optimal input sequence
        if config.MPC.initial_guess is None:
            self.initial_guess = 0.0 * self.std_dev
            self.best_control_vars = jnp.zeros((self.horizon,self.model.nu), dtype=self.dtype_general)
        else:
            self.initial_guess = config.MPC.initial_guess
            self.best_control_vars = jnp.tile(self.initial_guess, (self.horizon, 1))
        
        self.control_spline_grid = jnp.round(jnp.linspace(0, self.horizon, self.num_control_points)).astype(int).tolist()

        self.master_key = jax.random.PRNGKey(420)
        self.initial_random_parameters = jnp.zeros((self.num_parallel_computations, self.num_control_points, self.model.nu),
                                                   dtype=self.dtype_general)

        # Jit the controller function
        self.compute_control_mppi = jax.jit(self._compute_control_mppi, device=self.device)

        self.gains = jnp.zeros((model.nu, model.nx))
        # self.ctrl_sens_to_state = jax.jit(jax.jacfwd(self.compute_control_mppi, argnums=0, has_aux=True), device=self.device)
        self.rollout_sens_to_state = jax.vmap(jax.value_and_grad(self.rollout_single, argnums=0, has_aux=True), in_axes=(None, None, 0), out_axes=(0, 0))

        # Rename functions for cost during rollout
        self.cost_and_constraints = self.objective.cost_and_constraints
        self.final_cost_and_constraints = self.objective.final_cost_and_constraints
        
        

    
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def clip_input(self, control_variables):
        return jnp.clip(control_variables, self.input_min_full_horizon, self.input_max_full_horizon)

    def clip_input_single(self, control_variables):
        return jnp.clip(control_variables, self.input_min_full_horizon, self.input_max_full_horizon)
    
    # def clip_input_samples(self, control_variables):
    #     return jnp.clip(control_variables, self.input_min_full_horizon[self.control_spline_grid[1:], :], self.input_max_full_horizon[self.control_spline_grid[1:], :])

    @partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=(0, 0))
    def rollout_all(self, initial_state, reference, control_variables):
        return self.rollout_single(initial_state, reference, control_variables)
    
    def rollout_single(self, initial_state, reference, control_variables):
        
        cost = 0.0
        curr_state = initial_state
        # rollout_states = jnp.zeros((self.horizon+1, self.model.nx), dtype=self.dtype_general)
        # rollout_states = rollout_states.at[0, :].set(initial_state)
        if self.config.MPC.smoothing == "Spline":
            # control_variables = self.clip_input_samples(control_variables)
            control_interp = cubic_spline(self.control_spline_grid,
                                                        control_variables,
                                                        jnp.arange(0, self.horizon))
            control_variables = self.clip_input_single(control_interp)
        else:
            control_variables = self.clip_input_single(control_variables)

        # if self.config.MPC["augmented_reference"]:
        #     reference = reference.at[1:, -self.model.nu - 1:-1].set(control_variables)
        
        def cost_and_state_rollout(idx, cost_and_state):
            cost, curr_state = cost_and_state
            cost += self.dt*self.cost_and_constraints(curr_state, control_variables[idx, :], reference[idx, :])
            next_state = self.model.integrate_rollout_single(curr_state, control_variables[idx, :], self.dt)
            
            return cost, next_state

        cost, final_state = jax.lax.fori_loop(0, self.horizon, cost_and_state_rollout, (cost, curr_state))

        cost += self.dt*self.final_cost_and_constraints(final_state, reference[self.horizon, :])

        return cost, control_variables
    

    @partial(jax.vmap, in_axes=(None, None, None, 0, None), out_axes=(0, 0))
    def rollout_with_sensitivity(self, initial_state, reference, control_variables, mppi_gains):
        """
        Rollout of the system and associated parametric sensitivity dynamics
        :param initial_state:
        :param reference:
        :param control_variables:
        :param mppi_gains:
        :return:
        """
        cost = 0
        curr_state_sens = jnp.zeros((self.model.nx, self.model.np))
        curr_state = initial_state
        input_sequence = jnp.zeros((self.horizon, self.model.nu), dtype=self.dtype_general)
        if self.config.MPC["smoothing"] == "Spline":
            control_interp = cubic_spline(jnp.arange(0, self.horizon, self.control_points_sparsity),
                                        control_variables,
                                        jnp.arange(0, self.horizon))
            control_variables = self.clip_input_single(control_interp)

        for idx in range(self.horizon):
            curr_input = jax.lax.dynamic_slice(control_variables, (idx, 0), (1, self.model.nu)).reshape(-1)
            curr_input_sens = mppi_gains @ curr_state_sens
            cost_and_constraints = self.cost_and_constraints((curr_state, curr_state_sens), (curr_input, curr_input_sens), reference[idx, :])
            # Integrate the dynamics
            curr_state = self.model.integrate_rollout_single(curr_state[:self.model.nx], curr_input, self.dt)
            curr_state_sens = self.model.sensitivity_step(curr_state, curr_input, self.model.nominal_parameters, curr_state_sens, curr_input_sens, self.dt)
            cost += cost_and_constraints
            input_sequence = input_sequence.at[idx, :].set(curr_input)

        cost += self.final_cost_and_constraints((curr_state, curr_state_sens), reference[self.horizon, :])

        return cost, input_sequence

    def _compute_control_mppi(self, state, reference, best_control_vars, key, gains):
        additional_random_parameters = self.sample_input_sequence(key)

        if self.config.MPC.smoothing == "Spline":
            control_vars_all = best_control_vars[self.control_spline_grid, :] + additional_random_parameters
        else:
            control_vars_all = best_control_vars + additional_random_parameters

        # Do rollout
        if self.config.MPC.sensitivity:
            costs, control_vars_all = self.rollout_with_sensitivity(state, reference, control_vars_all, gains)
        else:
            if self.compute_gains:
                (costs, control_vars_all), gradients = self.rollout_sens_to_state(state, reference, control_vars_all)
            else:
                costs, control_vars_all = self.rollout_all(state, reference, control_vars_all)

        additional_random_parameters_clipped = (control_vars_all - best_control_vars)

        # Compute MPPI update
        costs, best_cost, worst_cost = self._sort_and_clip_costs(costs)
        # exp_costs = self._exp_costs_invariant(costs, best_cost, worst_cost)
        exp_costs = self._exp_costs_shifted(costs, best_cost)

        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom
        if self.compute_gains:
            weights_grad_shift = jnp.sum(weights[:, jnp.newaxis] * gradients, axis=0)
            weights_grad = -self.lam * weights[:, jnp.newaxis] * (gradients - weights_grad_shift)
            gains = jnp.sum(jnp.einsum('bi,bo->bio', weights_grad, additional_random_parameters_clipped[:, 0, :]), axis=0).T

        weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * additional_random_parameters_clipped
        optimal_action = best_control_vars + jnp.sum(weighted_inputs, axis=0)

        return optimal_action, gains


    @partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=(0, 0, 0, 0))
    def get_rollout(self, initial_state, reference, control_variables):
        lambda_ = 1
        cost = 0.0
        curr_state = initial_state
        if self.config.MPC.smoothing == "Spline":
            control_interp = cubic_spline(jnp.arange(0, self.horizon, self.control_points_sparsity),
                                                        control_variables,
                                                        jnp.arange(0, self.horizon))
            control_variables = self.clip_input_single(control_interp)
        else:
            control_variables = self.clip_input_single(control_variables)

        constraint_violation = 0
        for idx in range(self.horizon):
            cost += self.dt*self.cost_and_constraints(curr_state, control_variables[idx, :], reference[idx, :])
            curr_state = self.model.integrate_rollout_single(curr_state[:self.model.nx], control_variables[idx, :], self.dt)
            # l1_dist = jnp.abs(self.objective.constraints(curr_state, control_variables[idx, :], reference[idx, :])) 
            # constraint_violation += 1/jnp.sum(l1_dist)
            # dist_from_obs = self.objective.constraints_not_jit(curr_state, control_variables[idx, :], reference[idx, :])
            dist_from_obs = self.objective.constraints(curr_state, control_variables[idx, :], reference[idx, :])
            penalties = jnp.sum(jnp.where(dist_from_obs < 0.0, 1.0, 1e-5)) # close to 0 but not 0 to avoid log errors later on
            constraint_violation += penalties
        constraint_violation /= self.horizon
        cost += self.dt*self.final_cost_and_constraints(curr_state, reference[self.horizon, :])
        return cost, control_variables, initial_state, constraint_violation
 
    def get_rollouts(self, state, reference, num_steps=1):
        if reference.ndim == 1:
            reference = jnp.tile(reference, (self.horizon+1, 1))

        best_control_vars = self.best_control_vars
        rollouts = []
        for i in range(num_steps):
            additional_random_parameters = self.sample_input_sequence(self.master_key)

            if self.config.MPC.smoothing == "Spline":
                control_vars_all = best_control_vars[::self.control_points_sparsity, :] + additional_random_parameters
            else:
                control_vars_all = best_control_vars + additional_random_parameters

            costs, control_vars_all, inital_state, constraint_violation = self.get_rollout(state, reference, control_vars_all)
            # print(constraint_violation)
            rollouts.append([inital_state, control_vars_all, constraint_violation])

            additional_random_parameters_clipped = control_vars_all - best_control_vars
            costs, best_cost, worst_cost = self._sort_and_clip_costs(costs)
            exp_costs = self._exp_costs_shifted(costs, best_cost)
            denom = jnp.sum(exp_costs)
            weights = exp_costs / denom
            weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * additional_random_parameters_clipped
            optimal_action = best_control_vars + jnp.sum(weighted_inputs, axis=0)
            self._update_key()
      
        # self.best_control_vars = self._shift_guess(optimal_action)
    
        return rollouts, best_control_vars

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
            The optimal input trajectory shaped (horizon, nu)
        """
        # If the reference is just a state, repeat it along the horizon
        if reference.ndim == 1:
            reference = jnp.tile(reference, (self.horizon+1, 1))
        # if self.config.MPC["augmented_reference"]:
        #     reference = jnp.concatenate((reference, jnp.tile(self.last_input, (self.horizon+1, 1)), jnp.arange(0, self.horizon+1, 1).reshape(self.horizon+1, 1)), axis=1)

        best_control_vars = self.best_control_vars
        # maybe this loop should be jitted to actually be more efficient
        for i in range(num_steps):
            best_control_vars, gains = self.compute_control_mppi(state, reference, best_control_vars,
                                                                self.master_key, self.gains)
            self.gains = gains
            # Below are the gains computed using the full jax autodiff instead of the more efficient formula
            # print("old gains", self.ctrl_sens_to_state(state, reference, best_control_vars, self.master_key, self.gains)[0][0])
            self._update_key()

        self.last_input = best_control_vars[0]

        if shift_guess:
            self.best_control_vars = self._shift_guess(best_control_vars)
        else:
            self.best_control_vars = best_control_vars
        
        return best_control_vars
    
    def _sort_and_clip_costs(self, costs):
        # Saturate the cost in case of NaN or inf
        costs = jnp.where(jnp.isnan(costs), 1e6, costs)
        costs = jnp.where(jnp.isinf(costs), 1e6, costs)
        # Take the best found control parameters
        best_index = jnp.nanargmin(costs)
        worst_index = jnp.nanargmax(costs)
        best_cost = costs.take(best_index)
        worst_cost = costs.take(worst_index)

        return costs, best_cost, worst_cost

    def _exp_costs_shifted(self, costs, best_cost):
        return jnp.exp(- self.lam * (costs - best_cost))

    def _exp_costs_invariant(self, costs, best_cost, worst_cost):
        """
        For a comparison see:
        G. Rizzi, J. J. Chung, A. Gawel, L. Ott, M. Tognon and R. Siegwart,
        "Robust Sampling-Based Control of Mobile Manipulators for Interaction With Articulated Objects,"
        in IEEE Transactions on Robotics, vol. 39, no. 3, pp. 1929-1946, June 2023, doi: 10.1109/TRO.2022.3233343.

        Not used anymore ATM since it does not work with constraints (to be investigated)
        """
        h = 20.
        exp_costs = jnp.exp(- h * (costs - best_cost) / (worst_cost - best_cost))

        return exp_costs

    def _update_key(self):
        newkey, subkey = jax.random.split(self.master_key)
        self.master_key = newkey
    

    def sample_input_sequence(self, key):
        # Generate random parameters
        # The first control parameters is the old best one, so we add zero noise there
        additional_random_parameters = self.initial_random_parameters * 0.0
        # One sample is kept equal to the guess
        sampled_variation_all = jax.random.normal(key=key, shape=(self.num_parallel_computations-1, self.num_control_points, self.model.nu)) * self.std_dev

        additional_random_parameters = additional_random_parameters.at[1:, :, :].set(
            sampled_variation_all)

        return additional_random_parameters

class Controller:
    """
    Stateful controller calling the solver function and updating the its memory.
    
    This is needed to avoid triggering recompilation of the solver function at each iteration.
    """
    def __init__(self, solver):
        
        self.last_input = solver.initial_guess
        self.gains = solver.gains
        self.best_control_vars = solver.best_control_vars        
        
    def command(self, solver, state, reference, shift_guess=True, num_steps=1):
        
        # If the reference is just a state, repeat it along the horizon
        if reference.ndim == 1:
            reference = jnp.tile(reference, (solver.horizon+1, 1))

        best_control_vars = self.best_control_vars
        # maybe this loop should be jitted to actually be more efficient
        for i in range(num_steps):
            best_control_vars, gains = solver.compute_control_mppi(state, reference, best_control_vars,
                                                                solver.master_key, self.gains)
            self.gains = gains
            solver._update_key()

        self.last_input = best_control_vars[0]

        if shift_guess:
            self.best_control_vars = _shift_guess(best_control_vars)
        else:
            self.best_control_vars = best_control_vars
        
        return best_control_vars   
