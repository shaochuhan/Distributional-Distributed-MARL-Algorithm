# Distributional Distributed MARL Algorithm (D²MARL)
**Official PyTorch implementation of the paper: *Distributional Distributed MARL Algorithm***

## Overview
D²MARL is a fully decentralized multi-agent reinforcement learning (MARL) algorithm for **continuous action spaces** under communication constraints. It integrates distributional value estimation and Wasserstein policy optimization to address high policy gradient variance and heteroscedastic noise in fully distributed MARL systems.

The algorithm supports fully decentralized training with only local neighbor communication and provides theoretical convergence guarantees.

## Framework
![D²MARL Framework](https://github.com/shaochuhan/Distributional-Distributed-MARL-Algorithm/raw/master/d2marl-framework.jpg)

## Key Features
- **Distributional Critic**: Models return distribution via Gaussian parametrization; uses inverse-variance weighting from KL divergence for noise-robust learning.
- **Wasserstein Policy Optimization**: Stable policy update with zero sampling variance, combining exploration and stability.
- **Fully Decentralized**: No centralized controller; only local neighbor communication for consensus.
- **Theoretical Convergence**: Almost sure convergence to a stationary point under mild assumptions.
- **Scalable for Continuous Control**: Performs favorably on CACC, VMAS, and networked systems.

## Project Structure
```text
Distributional-Distributed-MARL-Algorithm/
├── agents/         # Core agent and trainer implementations
├── config/         # Hyperparameter configuration files
├── envs/           # Multi-agent environments
├── utils.py        # Utility functions
├── main.py         # Training and evaluation entry
└── requirements.txt # Dependencies
```

## Getting Started
### Train
```bash
python main.py --base-dir ./exp/d2marl train --config-dir ./config
```
### Evaluate
```bash
python main.py --base-dir ./exp/d2marl evaluate --evaluation-seeds 100
```
## Supported Environments
- Cooperative Adaptive Cruise Control (CACC)
- Vectorized Multi-Agent Simulator (VMAS)
- Networked System Control
