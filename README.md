# Multi-Strategy Rainbow DQN Implementation

**Author:** Mikey Ferguson
**Status:** Complete

## Overview

This repository contains an implementation of a **Rainbow DQN** agent for the 2025 RL Independent Study Final Project. The agent integrates features from three distinct categories of reinforcement learning advancements to solve discrete control environments.

The project performs a full ablation study (grid search) over 16 configurations ($2^4$) to isolate the effects of each technique on agent performance.

## The Power (Implemented Techniques)

This agent implements a "Rainbow" subset using **DQN** as the base algorithm.

### 1. Algorithm Specific Modifications

* **Distributional RL (C51):** The network outputs a probability distribution over value atoms rather than a single scalar Q-value.
* **Delayed Critic (Polyak Averaging):** Soft target network updates ($\tau \ll 1.0$) are used instead of standard hard updates to stabilize learning.

### 2. Algorithm Agnostic Features

* **KL-Penalty Term:** A regularization term penalizing the policy if it diverges too quickly from the target distribution.

### 3. Adversarial Modifications

* **Magnetic Mirror Descent (MMD):** A regularization term that tethers the online network weights to the target network weights, preventing catastrophic forgetting.

## Environments

The agent is evaluated on the following domains:

* **MiniGrid-FourRooms-v0:** A sparse-reward grid navigation task (Target Environment).
* **CartPole-v1:** Dense-reward control task (Sanity Check).
* **Leduc-v0 (via RLCard):** Discrete imperfect information poker.

## Project Structure

```text
.
├── launch_grid.ssub    # Slurm submission script for cluster training
├── logs/               # Slurm output and error logs
├── main.py             # Entry point: runs the ablation grid search
├── requirements.txt    # Python dependencies
├── src/
│   ├── agent.py        # RainbowAgent logic (C51 projection, loss calc)
│   ├── buffers.py      # ReplayBuffer implementation
│   ├── config.py       # Hyperparameters and AblationConfig definitions
│   ├── networks.py     # Neural Network architecture (Multi-head output)
│   └── wrappers.py     # Custom wrappers for RLCard (Leduc) and MiniGrid
└── results/            # Stores CSV logs and generated plots
````

## Installation

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Environment Dependencies (Crucial)
pip install minigrid rlcard
```

## Usage

### Local Execution (Testing)

To run the full grid search on your local machine (expect high CPU/GPU usage):

```bash
python -m main
```

*Note: Configure the active environment in `src/config.py` by modifying `HYPERPARAMETERS` keys before running.*

### Cluster Execution (The Real Training)

To submit the job to a Slurm cluster for the full 200,000+ frame training duration:

```bash
sbatch launch_grid.ssub
```

## Configuration

Hyperparameters are managed in `src/config.py`.

* **`AblationTechniques` Enum:** Defines the 16 binary permutations of the four implemented features.
* **`HYPERPARAMETERS` Dictionary:** Contains environment-specific settings (Learning Rate, Gamma, Buffer Size, Atom Count).

To enable or disable specific environments, comment/uncomment the relevant keys in `src/config.py`.

## Results

Training artifacts are saved to the `results/` directory.

* **Raw Data:** `raw_rewards.csv` and `smoothed_rewards.csv` for every configuration.
* **Visualization:** A combined plot showing the Exponential Moving Average (EMA) of all active techniques is generated automatically at the end of the run.
