# Sampling-Based MPC
A generic sampling-based MPC python library based on Jax.

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


# TODO
- [x] Control parametrization (splines etc)
- [x] Multivariate gaussian sampling 
- [x] Generic constraints
- [ ] adding Adam for interfacing the library with any urdf 


