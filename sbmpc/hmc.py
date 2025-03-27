import operator as op
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves, tree_reduce


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None: treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def normal_like_tree(rng_key, target, mean=0, std=1):
    # https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    keys_tree = random_split_like_tree(rng_key, target)
    return tree_map(lambda l, k: mean + std*jax.random.normal(k, l.shape, l.dtype), target, keys_tree)


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def leapfrog(params, momentum, log_prob_fn, step_size, n_steps):
    """Approximates Hamiltonian dynamics using the leapfrog algorithm."""

    # define a single step
    def step(i, args):
        params, momentum = args

        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum = tree_map(lambda m, g: m + 0.5 * step_size * g, momentum, grad)

        # update params
        params = tree_map(lambda p, m: p + m * step_size, params, momentum)

        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum = tree_map(lambda m, g: m + 0.5 * step_size * g, momentum, grad)

        return params, momentum

    # do 'n_steps'
    new_params, new_momentum = jax.lax.fori_loop(0, n_steps, step, (params, momentum))

    return new_params, new_momentum


def sample(key, params_init, log_prob_fn, n_steps, n_leapfrog_steps, step_size):
    """
    Runs HMC and returns the full Markov chain as a Pytree.
    - params: array
    - log_prob_fn: function that takes params as the only argument and returns a scalar value
    """

    # define a single step
    def step_fn(carry, x):
        params, key = carry
        key, normal_key, uniform_key = jax.random.split(key, 3)

        # generate random momentum
        momentum = normal_like_tree(key, params)

        # leapfrog
        new_params, new_momentum = leapfrog(params, momentum, log_prob_fn, step_size, n_leapfrog_steps)

        # MH correction
        potentaial_energy_diff = log_prob_fn(new_params) - log_prob_fn(params)
        momentum_dot = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum)))
        new_momentum_dot = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(new_momentum)))
        kinetic_energy_diff = 0.5*(momentum_dot - new_momentum_dot)
        log_accept_prob = potentaial_energy_diff + kinetic_energy_diff
        log_accept_prob = jnp.nan_to_num(log_accept_prob, nan=-jnp.inf)
        accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = ifelse(accept, new_params, params)

        return (params, key), (params, accept_prob)

    # do 'n_steps'
    _, (chain, accept_prob) = jax.lax.scan(step_fn, (params_init, key), xs=None, length=n_steps)
    
    # print(f'accept={accept_prob.mean():.2%}')
    return chain