import jax.numpy as jnp
import jax

from jax.scipy.signal import convolve


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


