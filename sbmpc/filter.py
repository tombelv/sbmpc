from functools import partial

import jax.numpy as jnp
import jax

from jax.scipy.signal import convolve
from math import factorial


class MovingAverage:
    def __init__(self, window_size: int = 3, step_size: int = 1):
        kernel = jnp.zeros(window_size + (window_size - 1) * (step_size - 1))
        for i in range(window_size):
            kernel = kernel.at[i*step_size].set(1)
        kernel = kernel.at[-1].set(1)
        self.kernel = kernel

        self.filter = jax.jit(self._filter)
        self.filter_batch = jax.vmap(self._filter, in_axes=(0, None, None))

        self.window_size = window_size

    def _filter(self, signal, left_padding, right_padding):
        signal_ = jnp.concatenate((left_padding, signal, right_padding))
        filtered = (convolve(signal_, self.kernel, mode='same') /
                    convolve(jnp.ones_like(signal_), self.kernel, mode='same'))
        return filtered[left_padding.shape[0]:(left_padding.shape[0] + signal.shape[0])]


# Savitzky-Golay that still has some bugs, to be tested
class SavitzkyGolay:
    def __init__(self, window_size: int = 3, step_size: int = 1):
        kernel = jnp.zeros(window_size + (window_size - 1) * (step_size - 1))
        for i in range(window_size):
            kernel = kernel.at[i*step_size].set(1)
        kernel = kernel.at[-1].set(1)
        self.kernel = kernel

        self.filter = jax.jit(self._filter)
        self.filter_batch = jax.vmap(self._filter, in_axes=(0, None, None))

        self.window_size = window_size

        order = 2
        deriv = 0
        rate = 1

        order_range = range(order + 1)
        self.half_window = (window_size - 1) // 2
        # precompute coefficients
        b = jnp.array([[k ** i for i in order_range] for k in range(-self.half_window, self.half_window + 1)])

        self.m = jnp.linalg.pinv(b)[deriv] * rate ** deriv * factorial(deriv)

    def _filter(self, y, left_padding, right_padding):
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - jnp.abs(y[1:self.half_window + 1][::-1] - y[0])
        lastvals = y[-1] + jnp.abs(y[-self.half_window - 1:-1][::-1] - y[-1])
        y_ = jnp.concatenate((firstvals, y, lastvals))
        return convolve(self.m[::-1], y_, mode='valid')


@partial(jax.vmap, in_axes=(None, 1, None), out_axes=1)
def cubic_spline_interpolation(x, y, x_new):
    """
    Perform cubic spline interpolation.

    Args:
    x: jnp.array, original x coordinates
    y: jnp.array, original y coordinates
    x_new: jnp.array, x coordinates to interpolate

    Returns:
    jnp.array, interpolated y values at x_new
    """
    n = x.shape[0]

    # Calculate the differences and weighted sums
    h = jnp.diff(x)
    a = jnp.diff(y) / h

    # Calculate the coefficients
    A = jnp.zeros((n, n))
    b = jnp.zeros(n)
    A = A.at[0, 0].set(1)
    A = A.at[-1, -1].set(1)

    def body_fun(i, val):
        A, b = val
        A = A.at[i, i - 1].set(h[i - 1])
        A = A.at[i, i].set(2 * (h[i - 1] + h[i]))
        A = A.at[i, i + 1].set(h[i])
        b = b.at[i].set(3 * (a[i] - a[i - 1]))
        return A, b

    A, b = jax.lax.fori_loop(1, n - 1, body_fun, (A, b))

    # Solve the tridiagonal system
    c = jnp.linalg.solve(A, b)

    # Compute the coefficients d and b
    d = jnp.diff(c) / (3 * h)
    b = a - h * (2 * c[:-1] + c[1:]) / 3

    # Perform interpolation
    def find_interval(x_i):
        return jnp.searchsorted(x, x_i, side='right') - 1

    def interpolate(x_i):
        idx = find_interval(x_i)
        dx = x_i - x[idx]
        return (y[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx])))

    return jax.vmap(interpolate)(x_new)

