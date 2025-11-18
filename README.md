# Sampling-Based MPC
A generic sampling-based MPC python library based on Jax.

Implements the Feedback-MPPI method presented in the [related paper](https://arxiv.org/abs/2506.14855) to compute a first order approximation of the MPPI solution suitable for high-frequency state feedback corrections.
```bibtex
@article{belvedere2025feedbackmppi,
      author={Belvedere, Tommaso and Ziegltrum, Michael and Turrisi, Giulio and Modugno, Valerio},
      title={Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation – Adios low-level controllers},
      journal={IEEE Robotics and Automation Letters},  
      year={2026},
      volume={11},
      number={1},
      pages={1-8},
      keywords={Robots;Trajectory;Costs;Real-time systems;Quadrupedal robots;Optimal control;Computational modeling;Standards;Legged locomotion;System dynamics;Optimization and Optimal Control;Motion Control;Legged Robots;Model Predictive Control},
      doi={10.1109/LRA.2025.3630871}
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


