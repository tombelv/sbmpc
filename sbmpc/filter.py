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