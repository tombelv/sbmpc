import jax.numpy as jnp
import jax
import numpy as np


@jax.jit
def skew(vec):
    return jnp.array([[0, -vec[2], vec[1]],
                      [vec[2], 0, -vec[0]],
                      [-vec[1], vec[0], 0]])


@jax.jit
def quat2rotm(quat):
    eta = quat[0]
    vec = quat[1:4]
    return jnp.array([[2*(eta * eta + vec[0] * vec[0]) - 1, 2*(vec[0] * vec[1] - eta * vec[2]), 2*(vec[0] * vec[2] + eta * vec[1])],
                      [2*(vec[0] * vec[1] + eta * vec[2]), 2*(eta * eta + vec[1] * vec[1]) - 1, 2*(vec[1] * vec[2] - eta * vec[0])],
                      [2*(vec[0] * vec[2] - eta * vec[1]), 2*(vec[1] * vec[2] + eta * vec[0]), 2*(eta * eta + vec[2] * vec[2]) - 1]])


@jax.jit
def quatProduct(quat1, quat2):
    vec1 = quat1[1:4]
    vec2 = quat2[1:4]

    # res = jnp.zeros(4, dtype=jnp.float32)
    # res.at[0] = quat1[0] * quat2[0] - vec1.dot(vec2)
    # res.at[1:4] = quat1[0] * vec2 + quat2[0]*vec1 + skew(vec1) @ vec2

    return jnp.concatenate([jnp.array([quat1[0] * quat2[0] - vec1.dot(vec2)]),
                            quat1[0] * vec2 + quat2[0]*vec1 + skew(vec1) @ vec2])

