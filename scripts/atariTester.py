"""
Test multiple pre-evolved agents in Atari environments

Run a certain number of episodes for each pre-evolved agent in parallel Atari environments,
show a tqdm progress bar, then plot each agent's reward distribution (with mean ± std and max) and save to a PNG.
"""

import os
import sys
import glob
import re
import yaml
import numpy as np
import gymnasium as gym
import ale_py
import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import argparse

from tqdm import tqdm
from SCOPE import *


# Hyperparameters and paths
WEIGHTS_ROOT          = "../visualization/out/best_solutions"
WEIGHTS_DIR           = "runs_5000_all"
GAME_NAME             = "ALE/SpaceInvaders-v5"
CONFIG_PATH           = "config.yaml"
TRIALS_PER_AGENT      = 1000
MAX_STEPS_PER_EPISODE = 10000

# Load the config file
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

OBS_TYPE           = config.get("OBS_TYPE", "grayscale")
REPEAT_ACTION_PROB = config.get("REPEAT_ACTION_PROBABILITY", 0)
FRAMESKIP          = config.get("FRAMESKIP", 4)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test multiple pre-evolved agents in Atari environments")
    parser.add_argument("--weights-dir", type=str, default="runs_5000_all", help="Directory name containing .npy solution files (default: runs_5000_all)")
    parser.add_argument("--trials-per-agent", type=int, default=1000, help="Number of episodes to run per agent (default: 1000)")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum steps per episode (default: 10000)")
    return parser.parse_args()


def parse_kp_from_filename(filename: str) -> tuple[int, float]:
    """
    Expect filenames like: <game_safe>_K<k>_P<p>_individual_<score>.npy
    and return the k and p values from the filename
    """
    parsed_values = re.search(r"_K(?P<k>\d+)_P(?P<p>\d+(?:\.\d+)?)_individual_\d+(?:\.\d+)?\.npy$", filename)
    if not parsed_values:
        raise ValueError(f"Cannot parse K/P from filename '{filename}'")
    return int(parsed_values.group("k")), float(parsed_values.group("p"))


def run_one_episode(agent_id: str,
                    weights_path: str,
                    k: int,
                    p: float,
                    max_steps: int) -> float:
    """Run a single episode with a given agent and return the total reward."""

    # Load the agent
    chromosome = np.load(weights_path)

    # Create the environment
    env = gym.make(id=GAME_NAME,
                   obs_type=OBS_TYPE,
                   repeat_action_probability=REPEAT_ACTION_PROB,
                   frameskip=FRAMESKIP)

    # Create the policy
    policy = SCOPE(chromosome=chromosome,
                   output_size=env.action_space.n,
                   k=k,
                   p=p)

    # Reset the environment
    obs, _ = env.reset()

    # Run the episode
    done, steps, ep_reward = False, 0, 0.0
    while not done and steps < max_steps:  
        state = obs.astype(np.float32) / 255.0
        prefs = policy.forward(state)
        action = int(np.argmax(prefs))
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        done = terminated or truncated
        steps += 1

    # Close the environment and return the episode reward
    env.close()
    return ep_reward


def run_one_task(task: tuple[str, str, int, float, int]) -> tuple[str, float]:
    """Unpack a task tuple that includes the agent_id, weights_path, k, p, and max_steps, and return (agent_id, reward)."""
    aid, wf, k, p, max_steps = task
    reward = run_one_episode(agent_id=aid,
                             weights_path=wf,
                             k=k,
                             p=p,
                             max_steps=max_steps)
    return aid, reward


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set parameters from arguments
    WEIGHTS_DIR = args.weights_dir
    TRIALS_PER_AGENT = args.trials_per_agent
    MAX_STEPS_PER_EPISODE = args.max_steps
    
    # Get the agent parameters from the filenames
    weight_files = sorted(glob.glob(os.path.join(WEIGHTS_ROOT, WEIGHTS_DIR, "*.npy")))
    agent_specs = []

    # For each weight file, get the k and p values from the filename and create an agent_id
    for wf in weight_files:
        try:
            k_val, p_val = parse_kp_from_filename(os.path.basename(wf))
        except ValueError:
            continue
        agent_id = f"K={k_val},P={p_val}"
        agent_specs.append((agent_id, wf, k_val, p_val))

    # If no valid .npy solution files are found, exit
    if not agent_specs:
        print(f"[ERROR] No valid .npy solution files found in {WEIGHTS_DIR}")
        sys.exit(1)

    # Prepare tasks and reward storage
    tasks = []
    all_rewards = {}
    for aid, wf, k, p in agent_specs:
        tasks.extend([(aid, wf, k, p, MAX_STEPS_PER_EPISODE)] * TRIALS_PER_AGENT)
        all_rewards[aid] = []

    # Run all trials in parallel with a tqdm progress bar
    num_cpus = mp.cpu_count()
    with mp.Pool(processes=num_cpus) as pool:
        for aid, reward in tqdm(pool.imap_unordered(run_one_task, tasks),
                                total=len(tasks),
                                desc="Running episodes"):
            all_rewards[aid].append(reward)

    # Plotting
    agent_ids = [aid for aid, *_ in agent_specs]
    n = len(agent_ids)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, aid in enumerate(agent_ids):
        ax = axes_flat[i]
        data = all_rewards[aid]
        mi, ma = int(min(data)) - 25, int(max(data)) + 25
        bins = list(range(mi, ma + 50, 50))
        ax.hist(data, bins=bins, edgecolor='black')
        mean, std, mx = np.mean(data), np.std(data), np.max(data)
        ax.text(0.95, 0.95,
                f"avg: {mean:.1f}±{std:.1f}\nmax: {mx:.1f}",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7, pad=0.3))
        ax.set_title(f"{aid}  (n={len(data)})")
        ax.set_yscale('log')

    # Remove any unused axes
    for ax in axes_flat[n:]:
        fig.delaxes(ax)

    fig.tight_layout(pad=3.0)
    fig.savefig("../visualization/out/atariTester_results.png")
    print("Done: saved reward distributions to results.png")


if __name__ == "__main__":
    main()
