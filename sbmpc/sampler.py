
from abc import ABC, abstractmethod
import jax

class Sampler(ABC):

    def __init__() -> None:
        pass

    def sample_input_sequence(self,key) -> None:
        pass

    def Update(self, new_parameters) -> None:
        pass

class GaussianSampler(Sampler):

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

class CEMSampler(Sampler):

    def __init__() -> None:
        pass

    def sample_input_sequence(self, key) -> None:
        pass