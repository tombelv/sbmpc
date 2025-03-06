
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp


class SBS(ABC):

    def __init__(self) -> None:
        self.best_control_vars=None

    def sample_input_sequence(self,key) -> None:
        pass

    def Update(self, parameters, costs) -> None:
        pass

class GaussianSampler(SBS):

    def __init__() -> None:
        pass

    def sample_input_sequence(self,key) -> None:
        # Generate random parameters
        # The first control parameters is the old best one, so we add zero noise there
        additional_random_parameters = self.initial_random_parameters * 0.0
        # One sample is kept equal to the guess
        sampled_variation_all = jax.random.normal(key=key, shape=(self.num_parallel_computations-1, self.num_control_points, self.model.nu)) * self.std_dev

        additional_random_parameters = additional_random_parameters.at[1:, :, :].set(
            sampled_variation_all)

        return additional_random_parameters
    def Update(self):
        pass

class CEMSampler(SBS):

    def __init__() -> None:
        super.__init()

    def sample_input_sequence(self, key):
        pass

    def Update(self,parameters, costs):


class MPPISampler(SBS):

    def __init__() -> None:
        super.__init()

    def sample_input_sequence(self, key):
        pass

    def Update(self,parameters, costs):
        # Compute MPPI update
        costs, best_cost, worst_cost = self._sort_and_clip_costs(costs)
        # exp_costs = self._exp_costs_invariant(costs, best_cost, worst_cost)
        exp_costs = self._exp_costs_shifted(costs, best_cost)

        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom
        

        weighted_inputs = weights[:, jnp.newaxis, jnp.newaxis] * additional_random_parameters_clipped
        optimal_action = self.best_control_vars + jnp.sum(weighted_inputs, axis=0)

    def _sort_and_clip_costs(cost):
        pass

    def _exp_costs_shifted(cost,best_cost):
        pass



