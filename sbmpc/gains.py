# here i need to make an interface class to compute gains 
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from sbmpc.settings import Config

from functools import partial

class Gains(ABC):

    def __init__(self, config: Config) -> None:
        self.compute_gains = config.MPC.gains
        self.lam = config.MPC.lambda_mpc
        self.cur_gains =  jnp.zeros((config.robot.nu, config.robot.nx))
        

    @abstractmethod
    def gains_computation(self,key) -> jnp.ndarray:
        pass





class MPPIGain(Gains):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @partial(jax.jit, static_argnums=(0,))
    def gains_computation(self, costs, samples_delta, gradients) -> jnp.ndarray:
        if self.compute_gains:
            # Compute update for weights
            costs, best_cost, worst_cost = self._sort_and_clip_costs(costs)
            # exp_costs = self._exp_costs_invariant(costs, best_cost, worst_cost)
            exp_costs = self._exp_costs_shifted(costs, best_cost)
            denom = jnp.sum(exp_costs)
            weights = exp_costs / denom
            weights_grad_shift = jnp.sum(weights[:, jnp.newaxis] * gradients, axis=0)
            weights_grad = -self.lam * weights[:, jnp.newaxis] * (gradients - weights_grad_shift)
            gains = jnp.sum(jnp.einsum('bi,bo->bio', weights_grad, samples_delta[:, 0, :]), axis=0).T
        else:
            # if im not computing the gains i return the initial gains which are all zeros
            gains = self.cur_gains
        return gains 

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
