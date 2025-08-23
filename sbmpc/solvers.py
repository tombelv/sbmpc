from sbmpc.model import BaseModel
from sbmpc.settings import Config
from sbmpc.sampler import Sampler
from sbmpc.gains import Gains

import jax.numpy as jnp
import jax
import numpy as np

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
        return self.running_cost(state, inputs, reference) + jnp.sum(self.make_barrier(self.constraints(state, inputs, reference)[0]))

    def final_cost_and_constraints(self, state, reference):
        return self.final_cost(state, reference) + jnp.sum(self.make_barrier(self.terminal_constraints(state, reference)))

    def make_barrier(self, constraint_array):
        constraint_array = jnp.where(constraint_array < 0, 1e3, 0.0)
        return constraint_array

    def constraints(self, state, inputs, reference):
        return 0.0

    def terminal_constraints(self, state, reference):
        return 0.0

    def write(self,num_rej):
        self.num_sample_rejections.append(num_rej)
        f = open("/home/ubuntu/sbmpc/experiments/sample_rejections.txt", "w")
        f.write(str(np.array(self.num_sample_rejections)))
        f.close()
        return num_rej
    


class RolloutGenerator():

    def __init__(self, model: BaseModel, objective: BaseObjective, config: Config):
        """
        Initializes the rollout generator with the model, the objective, configurations and initial guess.
        Parameters
        ----------
        model: BaseModel
            The model propagated during rollouts.
        objective: BaseObjective
            Required to compute the cost function in the rollout.
        config_mpc: ConfigMPC
            Contains the MPC related parameters such as the time horizon, number of samples, etc.
        """

        self.dtype_general = config.general.dtype
        self.device = config.general.device

        self.model = model
        self.objective = objective

        self.config = config

        # Sampling time for discrete time model
        self.dt = config.MPC.dt
        # Control horizon of the MPC (steps)
        self.horizon = config.MPC.horizon
        # Monte-carlo samples, that is the number of trajectories that are evaluated in parallel 
        # check if we need to move it
        self.num_parallel_computations = config.MPC.num_parallel_computations


        self.compute_gains = config.MPC.gains
        
        # Covariance of the input action
        # self.sigma_mppi = jnp.diag(config.MPC.std_dev_mppi**2)
  
        self.num_control_points = config.MPC.num_control_points
        self.control_points_sparsity = self.horizon // self.num_control_points

        self.input_max_full_horizon = jnp.tile(model.input_max, (self.horizon, 1))
        self.input_min_full_horizon = jnp.tile(model.input_min, (self.horizon, 1))
        self.clip_input = jax.jit(self.clip_input, device=self.device)

        self.control_spline_grid = jnp.round(jnp.linspace(0, self.horizon, self.num_control_points)).astype(int).tolist()

        #self.gains = jnp.zeros((model.nu, model.nx))
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
    

    @partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=(0, 0, 0))
    def rollout_all(self, initial_state, reference, control_variables):
        if self.config.MPC.sensitivity:
            return self.rollout_single_with_sensitivity(initial_state, reference, control_variables)
        else:
            return self.rollout_single(initial_state, reference, control_variables)
    
    def interpolate_control(self, control_variables):
        """"
        Interpolates the control variables over the full horizon or passes the control variables directly
        """
        if self.config.MPC.smoothing == "Spline":
            control_interp = cubic_spline(self.control_spline_grid, control_variables, jnp.arange(0, self.horizon))
            return self.clip_input_single(control_interp)
        else:
            return self.clip_input_single(control_variables)
    
    def rollout_single(self, initial_state, reference, control_variables):
        cost = 0.0
        curr_state = initial_state
        rejections_total = 0

        control_variables = self.interpolate_control(control_variables)
        
        def cost_and_state_rollout(idx, cost_and_state):
            cost, curr_state, _ = cost_and_state
            cost += self.dt*self.cost_and_constraints(curr_state, control_variables[idx, :], reference[idx, :])
            next_state = self.model.integrate_rollout_single(curr_state, control_variables[idx, :], self.dt)
            rejections = jnp.sum(jnp.where((self.objective.constraints(curr_state, control_variables[idx, :], reference[idx, :])[0]) < 0, 1, 0))  
            
            return cost, next_state, rejections

        cost, final_state, rejections = jax.lax.fori_loop(0, self.horizon, cost_and_state_rollout, (cost, curr_state, rejections_total))
        rejections_total += rejections

        cost += self.dt*self.final_cost_and_constraints(final_state, reference[self.horizon, :])

        return cost, control_variables, rejections_total
    

    def rollout_single_with_sensitivity(self, initial_state, reference, control_variables):
        cost = 0.0
        curr_state = initial_state
        curr_state_sens = jnp.zeros((self.model.nx, self.model.np))

        control_variables = self.interpolate_control(control_variables)
        
        def cost_and_state_rollout(idx, cost_and_state):
            cost, curr_state = cost_and_state
            cost += self.dt*self.cost_and_constraints(curr_state, control_variables[idx, :], reference[idx, :])
            next_state = self.model.integrate_rollout_single(curr_state, control_variables[idx, :], self.dt)
            
            return cost, next_state

        cost, final_state = jax.lax.fori_loop(0, self.horizon, cost_and_state_rollout, (cost, curr_state))

        cost += self.dt*self.final_cost_and_constraints(final_state, reference[self.horizon, :])

        return cost, control_variables
    

    # TODO UPDATE
    # @partial(jax.vmap, in_axes=(None, None, None, 0, None), out_axes=(0, 0))
    # def rollout_with_sensitivity(self, initial_state, reference, control_variables, mppi_gains):
    #     """
    #     Rollout of the system and associated parametric sensitivity dynamics
    #     :param initial_state:
    #     :param reference:
    #     :param control_variables:
    #     :param mppi_gains:
    #     :return:
    #     """
    #     cost = 0
    #     curr_state_sens = jnp.zeros((self.model.nx, self.model.np))
    #     curr_state = initial_state
    #     input_sequence = jnp.zeros((self.horizon, self.model.nu), dtype=self.dtype_general)
    #     if self.config.MPC["smoothing"] == "Spline":
    #         control_interp = cubic_spline(jnp.arange(0, self.horizon, self.control_points_sparsity),
    #                                     control_variables,
    #                                     jnp.arange(0, self.horizon))
    #         control_variables = self.clip_input_single(control_interp)

    #     for idx in range(self.horizon):
    #         curr_input = jax.lax.dynamic_slice(control_variables, (idx, 0), (1, self.model.nu)).reshape(-1)
    #         curr_input_sens = mppi_gains @ curr_state_sens
    #         cost_and_constraints = self.cost_and_constraints((curr_state, curr_state_sens), (curr_input, curr_input_sens), reference[idx, :])
    #         # Integrate the dynamics
    #         curr_state = self.model.integrate_rollout_single(curr_state[:self.model.nx], curr_input, self.dt)
    #         curr_state_sens = self.model.sensitivity_step(curr_state, curr_input, self.model.nominal_parameters, curr_state_sens, curr_input_sens, self.dt)
    #         cost += cost_and_constraints
    #         input_sequence = input_sequence.at[idx, :].set(curr_input)

    #     cost += self.final_cost_and_constraints((curr_state, curr_state_sens), reference[self.horizon, :])

    #     return cost, input_sequence

    @partial(jax.jit, static_argnums=(0,))  
    def do_rollout(self, state, reference, optimal_samples, samples_delta, gains):
        gradients = None

        if self.config.MPC.smoothing == "Spline":
            control_vars_all = optimal_samples[self.control_spline_grid, :] + samples_delta
        else:
            control_vars_all = optimal_samples + samples_delta

        # If the reference is just a state, repeat it along the horizon
        if reference.ndim == 1:
            reference = jnp.tile(reference, (self.horizon+1, 1))

        if self.compute_gains:
            (costs, control_vars_all), gradients = self.rollout_sens_to_state(state, reference, control_vars_all)
        else:
            costs, control_vars_all, rejections_all = self.rollout_all(state, reference, control_vars_all)

        samples_delta_clipped = self.compute_samples_delta(control_vars_all, optimal_samples)

        return samples_delta_clipped, costs, gradients, jnp.sum(rejections_all)
    

    def compute_samples_delta(self, control_action, optimal_samples):
        samples_delta_clipped = (control_action - optimal_samples)
        return samples_delta_clipped
    
    
class Controller:
    def __init__(self, rollout_gen : RolloutGenerator, sampler : Sampler, gains_obj: Gains):

        self.rollout_gen = rollout_gen
        self.sampler = sampler
        self.gains_obj = gains_obj
           
    def command(self, state, reference, shift_guess=True, num_steps=1):
        optimal_samples = self.sampler.optimal_samples
        gains = self.gains_obj.cur_gains

        for i in range(num_steps):
            samples_delta = self.sampler.sample_input_sequence(self.sampler.master_key, state) # generate random samples
            samples, costs, gradients, rejections = self.rollout_gen.do_rollout(state, reference, optimal_samples, samples_delta, gains) # rollout i.e. get cost of those samples
            optimal_samples = self.sampler.update(optimal_samples, samples, costs, rejections) # weight with cost to get optimal samples i.e. effectively reject costly samples
            # update gains
            self.gains_obj.cur_gains = self.gains_obj.gains_computation(costs, samples, gradients)
       
        # update sampler best control vars
        if shift_guess:
            self.sampler.optimal_samples = self._shift_guess(optimal_samples)
        else:
            self.sampler.optimal_samples = optimal_samples
        
        return optimal_samples # i.e. input sequence
    

    @partial(jax.jit, static_argnums=(0,))
    def _shift_guess(self, optimal_samples):
        optimal_samples_shifted = jnp.roll(optimal_samples, shift=-1, axis=0)
        optimal_samples_shifted = optimal_samples_shifted.at[-1, :].set(
            optimal_samples_shifted[-2:-1, :].reshape(-1))
        return optimal_samples_shifted
    
    @partial(jax.vmap, in_axes=(None, None, None, None, 0), out_axes=(0, 0, 0, 0))
    def get_rollout(self, initial_state, reference, optimal_samples, samples_delta): # do rollout + rollout all + rollout single
        cost = 0.0
        constraint_violation = 0.0

        if self.rollout_gen.config.MPC.smoothing == "Spline":
            control_vars = optimal_samples[self.rollout_gen.control_spline_grid, :] + samples_delta
        else:
            control_vars = optimal_samples + samples_delta

        current_state = initial_state
        final_state = initial_state

        control_vars = self.rollout_gen.interpolate_control(control_vars)

        for idx in range(self.rollout_gen.horizon):
            cost += self.rollout_gen.dt * self.rollout_gen.cost_and_constraints(current_state, control_vars[idx, :], reference[idx, :])
            final_state = self.rollout_gen.model.integrate_rollout_single(current_state, control_vars[idx, :], self.rollout_gen.dt)

            close_to_obs = self.rollout_gen.objective.constraints_not_jit(current_state, control_vars[idx, :], reference[idx, :])
            penalties = jnp.sum(jnp.array(close_to_obs)) + 1e-5 # avoid divide by 0 errors when scaling later
            constraint_violation += penalties
            
        cost += self.rollout_gen.dt*self.rollout_gen.final_cost_and_constraints(final_state, reference[self.rollout_gen.horizon, :])
        constraint_violation /= self.rollout_gen.horizon
        samples_delta_clipped = self.rollout_gen.compute_samples_delta(control_vars, optimal_samples)

        return samples_delta_clipped, cost, initial_state, constraint_violation
    
    def get_rollouts(self, state, reference, shift_guess=True, num_steps=1): # command
        optimal_samples = self.sampler.optimal_samples

        rollouts = []
        for i in range(num_steps):
            samples_delta = self.sampler.sample_input_sequence(self.sampler.master_key)
            samples, costs, initial_state, constraint_violation = self.get_rollout(state, reference, optimal_samples, samples_delta) 
            optimal_samples = self.sampler.update(optimal_samples, samples, costs, state)
            rollouts.append([initial_state, samples, constraint_violation])

        if shift_guess:
            self.sampler.optimal_samples = self._shift_guess(optimal_samples)
        else:
            self.sampler.optimal_samples = optimal_samples
        
        return rollouts, optimal_samples
