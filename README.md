# Sparse Cosine Optimized Policy Evolution for Atari Games

[Jim O'Connor](https://oconnor.digital.conncoll.edu) | [Jay B. Nash](https://www.linkedin.com/in/jaybnash/) | [Derin Gezgin](https://deringezgin.github.io) | [Gary B. Parker](https://oak.conncoll.edu/parker/)

*Published in AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, 2025*

## Overview

Sparse Cosine Optimized Policy Evolution (**SCOPE**) is a lightweight reinforcement-learning framework that evolves compact policies for Atari 2600 games.  
The core idea is to compress each frame with a 2-D Discrete Cosine Transform (DCT), keep only a small **k×k** block of low-frequency coefficients, sparsify it by zeroing the lowest **p-th** percentile, and map the result to actions through two linear layers.  
Policy parameters are optimized end-to-end with **Covariance-Matrix Adaptation Evolution Strategy (CMA-ES)**, allowing training to remain gradient-free, highly parallel, and easy to scale across CPUs.

---

## Key Features

* **Compact policies** – order-of-magnitude fewer parameters than convolutional networks.
* **Gradient-free optimisation** using CMA-ES; no back-prop or differentiable emulator needed.
* **Ray-based distributed execution** for embarrassingly parallel evaluation on clusters.
* **Turn-key parameter sweep** and multi-game benchmarking utilities.
* **Rich visualisation scripts** for learning curves, heat-maps, and best-agent roll-outs.

---

## Repository Structure

```
SCOPE-for-Atari/
├── data/                   # SQLite DBs & helper utilities
├── scripts/                # Training & evaluation entry-points
│   ├── SCOPE.py            # Core policy definition
│   ├── single_run.py       # Minimal example run
│   ├── parameter_sweep.py  # Multiple across K & P and different games
│   ├── atariTester.py      # Evaluate pre-evolved agents
│   ├── ray_setup.py        # Helper to bootstrap a Ray cluster via SSH
│   └── config.yaml         # Default hyper-parameters
├── visualization/          # Plotting utilities
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Installation

**Clone the repo**

```bash
git clone https://github.com/ConnAALL/SCOPE-for-Atari.git
cd SCOPE-for-Atari
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

Python ≥ 3.9 is recommended.

---

## Quick Start

### 1. Train on a single game

```bash
python scripts/single_run.py
```

The default configuration (`scripts/config.yaml`) will optimize a SCOPE policy on *Space Invaders* for 5 000 generations and print the training information.

### 2. Parameter sweep / multi-game benchmark

```bash
# Full factorial sweep over K and P ranges on one game
python scripts/parameter_sweep.py --mode full-sweep

# Partial sweep with specific parameters on one game
python scripts/parameter_sweep.py --mode partial-sweep

# Multi-Game Parameter Sweep
python scripts/parameter_sweep.py --mode multi_game-sweep
```

Results (fitness logs, best individuals, learning-curve data) are kept entirely in-memory and returned as a Python object, and saved to an SQLite database for later analysis.

### 3. Visualise learning curves & best agents

After completing the training runs, there are multiple visualization options. 

```bash
# Aggregate and convert DB runs to .npy solutions, and visualize statistics
python visualization/aggregate_best_fitness.py --db-path data/runs_5000_all.db

# Plot averaged curves for a DB
python visualization/plot_averaged_curves.py --db-path data/runs_5000_all.db

# Heat-map of K/P sweeps
python visualization/heatmap.py
```

### 4. Test evolved agents

```bash
python scripts/atariTester.py \
  --weights-dir runs_5000_all \
  --trials-per-agent 1000 \
  --max-steps 10000
```

A progress-bar is shown; afterwards a PNG with reward distributions for each agent is written to `visualization/out/`.

---

## Distributed Training with Ray

To scale evaluation across multiple machines, first start running a Ray cluster:

```bash
python scripts/ray_setup.py --username <ssh_user> --password <ssh_pass>
```

Edit the list of `worker_hosts` in `ray_setup.py` to match your infrastructure.  Once the cluster is ready, simply run `parameter_sweep.py`; each evaluation task is annotated with `@ray.remote` and will be scheduled across available CPUs.

---

## Configuration

All tunable hyper-parameters are collected in `scripts/config.yaml` and loaded by the training scripts.  Command-line flags or the override dictionaries (see `parameter_sweep.py`) can be used to patch any field at runtime.

Key fields:

* `ENV_NAME` – Gymnasium ID of the Atari environment.
* `K`, `P` – SCOPE hyperparameters.
* `CMA_SIGMA`, `POPULATION_SIZE`, `GENERATIONS` – CMA-ES settings.
* `EPISODES_PER_INDIVIDUAL`, `MAX_STEPS_PER_EPISODE` – evaluation parameters

---

## Citation

```bibtex
@inproceedings{10.1609/aiide.v21i1.36834,
author = {O'Connor, Jim and Nash, Jay B. and Gezgin, Derin and Parker, Gary B.},
title = {Playing atari space invaders with sparse cosine optimized policy evolution},
year = {2025},
isbn = {1-57735-904-6},
publisher = {AAAI Press},
url = {https://doi.org/10.1609/aiide.v21i1.36834},
doi = {10.1609/aiide.v21i1.36834},
abstract = {Evolutionary approaches have previously been shown to be effective learning methods for a diverse set of domains. However, the domain of game-playing poses a particular challenge for evolutionary methods due to the inherently large state space of video games. As the size of the input state expands, the size of the policy must also increase in order to effectively learn the temporal patterns in the game space. Consequently, a larger policy must contain more trainable parameters, exponentially increasing the size of the search space. Any increase in search space is highly problematic for evolutionary methods, as increasing the number of trainable parameters is inversely correlated with convergence speed. To reduce the size of the input space while maintaining a meaningful representation of the original space, we introduce Sparse Cosine Optimized Policy Evolution (SCOPE). SCOPE utilizes the Discrete Cosine Transform (DCT) as a pseudo attention mechanism, transforming an input state into a coefficient matrix. By truncating and applying sparsification to this matrix, we reduce the dimensionality of the input space while retaining the highest energy features of the original input. We demonstrate the effectiveness of SCOPE as the policy for the Atari game Space Invaders. In this task, SCOPE with CMA-ES outperforms evolutionary methods that consider an unmodified input state, such as OpenAI-ES and HyperNEAT. SCOPE also outperforms simple reinforcement learning methods, such as DQN and A3C. SCOPE achieves this result through reducing the input size by 53\% from 33,600 to 15,625 then using a bilinear affine mapping of sparse DCT coefficients to policy actions learned by the CMA-ES algorithm. The results presented in this paper demonstrate that the use of SCOPE allow evolutionary computation to achieve results competitive with reinforcement learning methods and far beyond what previous evolutionary methods have achieved.},
booktitle = {Proceedings of the Twenty-First AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
articleno = {31},
numpages = {10},
location = {Edmonton, Alberta, Canada},
series = {AIIDE '25}
}
```
