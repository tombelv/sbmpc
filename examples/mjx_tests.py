import time, os

import jax
import jax.numpy as jnp

os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
    )

import matplotlib.pyplot as plt

from sbmpc.model import Model, ModelMjx
from sbmpc.utils.geometry import skew, quat_product, quat2rotm, quat_inverse

input_max = jnp.array([1, 2.5, 2.5, 2])
input_min = jnp.array([0, -2.5, -2.5, -2])

mass = 0.027
gravity = 9.81
inertia = jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5], dtype=jnp.float32)
inertia_mat = jnp.diag(inertia)

spatial_inertia_mat = jnp.diag(jnp.concatenate([mass*jnp.ones(3, dtype=jnp.float32), inertia]))
spatial_inertia_mat_inv = jnp.linalg.inv(spatial_inertia_mat)

input_hover = jnp.array([mass*gravity, 0., 0., 0.], dtype=jnp.float32)


@jax.jit
def quadrotor_dynamics(state: jnp.array, inputs: jnp.array) -> jnp.array:
    """
    Simple quadrotor dynamics model with CoM placed at the geometric center

    Parameters
    ----------
    state : jnp.array
        state vector [pos (world frame),
                      attitude (unit quaternion [w, x, y, z]),
                      vel (world frame),
                      angular_velocity (body frame)]
    inputs : jnp.array):
        input vector [thrust (along the body-frame z axis), torque (body frame)]
    Returns
    -------
    state_dot :jnp.array
        time derivative of state with given inputs
    """

    quat = state[3:7]
    ang_vel = state[10:13]

    orientation_mat = quat2rotm(quat)
    ang_vel_quat = jnp.array([0., state[10], state[11], state[12]])

    total_force = jnp.array([0., 0., inputs[0]]) - mass*gravity*orientation_mat[2, :]  # transpose + 3rd col = 3rd row

    total_torque = 1e-3*inputs[1:4] - skew(ang_vel) @ inertia_mat @ ang_vel  # multiplication by normalization factor

    acc = spatial_inertia_mat_inv @ jnp.concatenate([total_force, total_torque])

    state_dot = jnp.concatenate([state[7:10],
                                 0.5 * quat_product(quat, ang_vel_quat),
                                 orientation_mat @ acc[:3],
                                 acc[3:6]])

    return state_dot


model_classic = Model(quadrotor_dynamics, 7, 6, 4, [input_min, input_max])

model_mjx = ModelMjx("bitcraze_crazyflie_2/cf2.xml")

q_init = jnp.array([0.0, 0.0, 0., 1., 0., 0., 0.], dtype=jnp.float32)  # hovering position

x_init = jnp.concatenate([q_init, jnp.zeros(model_classic.nv, dtype=jnp.float32)], axis=0)

# integ_classic_batch = jax.jit(jax.vmap(model_classic.integrate, in_axes=(None, 0, None)))
integ_mjx_batch = jax.jit(jax.vmap(model_mjx.integrate, in_axes=(None, 0, None)))

batch_sizes = [10, 100, 1000, 10000, 100000]


for batch_size in batch_sizes:
    state_batch = jnp.tile(x_init, (batch_size, 1))
    input_batch = jax.random.uniform(jax.random.PRNGKey(0), shape=(batch_size, 4), dtype=jnp.float32, minval=-1.,
                                     maxval=1.)

    jax.block_until_ready(model_classic.integrate_rollout(state_batch, input_batch, 0.02))
    jax.block_until_ready(model_mjx.integrate_rollout(state_batch, input_batch, 0.02))
    jax.block_until_ready(integ_mjx_batch(model_mjx.data, input_batch, 0.02))

    time_start = time.time_ns()
    jax.block_until_ready(model_classic.integrate_rollout(state_batch, input_batch, 0.02))
    comp_time = 1e-6 * (time.time_ns() - time_start)
    print(f"classic integration for a batch size = {batch_size}: {comp_time:.3f} [ms]")

    time_start = time.time_ns()
    jax.block_until_ready(model_mjx.integrate_rollout(state_batch, input_batch, 0.02))
    comp_time = 1e-6 * (time.time_ns() - time_start)
    print(f"mjx integration for a batch size = {batch_size}: {comp_time:.3f} [ms]")

    time_start = time.time_ns()
    jax.block_until_ready(integ_mjx_batch(model_mjx.data, input_batch, 0.02))
    comp_time = 1e-6 * (time.time_ns() - time_start)
    print(f"mjx integration for a batch size = {batch_size}: {comp_time:.3f} [ms]")

