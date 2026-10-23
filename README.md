# Morphing Rovers Optimization

University project for Algorithms and Complexity.

## Project Description

This project investigates optimization algorithms for the
ESA SpOC-2 Morphing Rovers challenge.

The objective is to optimize a rover's terrain-adaptive
forms and neural network parameters to improve navigation
performance across 30 simulation scenarios.

## Optimization Methods

- Baseline Random Mutation Search
- TAMRO (Terrain-Adaptive Morphing Rover Optimization)
- TAMRO v2 (Quick Evaluation)
- Scenario Performance Analysis

## Current Results

| Method | Fitness |
|---|---|
| ESA Example | 7.931893 |
| Baseline Optimizer | 7.790642 |
| TAMRO v1 | 6.982332 |

Lower fitness values indicate better performance.

## Technologies

- Python 3.12
- NumPy
- PyTorch
- Torchvision
- Matplotlib
- ImageIO

## Reference

ESA SpOC-2:
https://github.com/esa/SpOC2

## Development Status

Work in progress. Further optimization, experiments,
documentation and reproducibility improvements are planned.

The ESA evaluation code and datasets are not included
in this repository yet.
