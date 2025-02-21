# Sampling-Based MPC
A generic sampling-based MPC python library based on jax.

# Installation
## Requirements
 - [Nvidia cuda toolkit](https://developer.nvidia.com/cuda-toolkit) installed system-wide
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
```
pip install -e .

pip install -e ".[cuda12]"

pip install -e ".[cuda12_local]"

```

# TODO
- [x] Control parametrization (splines etc)
- [x] Multivariate gaussian sampling 
- [x] Generic constraints
- [ ] adding Adam for interfacing the library with any urdf 


