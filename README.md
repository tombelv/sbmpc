# Sampling-Based MPC
A generic sampling-based MPC python library based on Jax.

Implements the Feedback-MPPI method presented in the [related paper](https://arxiv.org/abs/2506.14855) to compute a first order approximation of the MPPI solution suitable for high-frequency state feedback corrections.
```bibtex
@article{belvedere2026feedbackmppi,
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
 - [pixi](https://pixi.sh/) - A fast package manager built on top of conda

## Instructions

### CPU-only installation (default)
Install dependencies and activate the CPU-only environment:
```bash
pixi install
pixi shell
```

### CUDA-enabled installation
For GPU acceleration with CUDA support:
```bash
pixi install -e cuda
pixi shell -e cuda
```

### Building the package
To build the Python package:
```bash
pixi run build
```

Or with a specific environment:
```bash
pixi run -e cuda build
```

### Running examples
Run examples directly with pixi:
```bash
pixi run python examples/quadrotor.py
```

Or with the CUDA environment:
```bash
pixi run -e cuda python examples/quadrotor.py
```

Refer to the [Jax documentation](https://jax.readthedocs.io/) for more details on GPU acceleration.


## Contributors

- Tommaso Belvedere, CNRS (core developer, project lead)
- Michael Ziegltrum, UCL (feature developer)
- Chidinma Ezeji, UCL (feature developer)
- Giulio Turrisi, IIT (project lead)
- Valerio Modugno, UCL (core developer, project lead)


