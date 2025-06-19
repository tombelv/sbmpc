# Sampling-Based MPC
A generic sampling-based MPC python library based on Jax.

Implements the Feedback-MPPI method presented in the [related paper](https://arxiv.org/abs/2506.14855) to compute a first order approximation of the MPPI solution suitable for high-frequency state feedback corrections.
```
@article{belvedere2025feedbackmppi,
      title={Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation -- Adios low-level controllers}, 
      author={Tommaso Belvedere and Michael Ziegltrum and Giulio Turrisi and Valerio Modugno},
      year={2025},
      url={https://arxiv.org/abs/2506.14855}, 
}
```

# Installation
## Requirements
 - [Nvidia cuda toolkit](https://developer.nvidia.com/cuda-toolkit) installed system-wide (if you want to use a local CUDA version)
 - [miniforge](https://github.com/conda-forge/miniforge/releases)

## Instructions
Create the conda environment with
```
mamba env create -f environment.yml
```

Activate the environment with
```
conda activate sbmpc
```

Depending on the CUDA settings of your machine, choose between
- CPU-only acceleration
```
pip install -e .
```
- GPU acceleration with pip-installed CUDA libraries
```
pip install -e ".[cuda12]"
```
- GPU acceleration with locally installed CUDA
```
pip install -e ".[cuda12_local]"
```

Refer to the Jax documentation for details.


## Contributors

- Tommaso Belvedere, CNRS (core developer, project lead)
- Michael Ziegltrum, UCL (feature developer)
- Chidinma Ezeji, UCL (feature developer)
- Giulio Turrisi, IIT (project lead)
- Valerio Modugno, UCL (core developer, project lead)


