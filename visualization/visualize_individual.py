"""
Visualize the best individual for a given game and hyperparameters

It creates a GIF of the best individual for a given game and hyperparameters.
The GIF shows the grayscale input, the DCT, the sparse DCT, and the action logits.
"""

import os
import json
import sqlite3
import argparse
import sys

import numpy as np
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.animation import PillowWriter
from scipy.fftpack import dct

# Add the scripts directory to the path to import SCOPE
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.SCOPE import *


def get_best_solution(db_path: str,
                      game: str,
                      k: int,
                      p: float) -> np.ndarray:
    """Get the best solution from the database"""

    # Resolve database path relative to the project root
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', db_path)

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = """
        SELECT best_individuals_json
        FROM runs
        WHERE game = ?
        AND overrides_json = ?
        ORDER BY best_fitness DESC
        LIMIT 1
    """
    cursor.execute(query, (game, json.dumps({"K": k, "P": p})))
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"No run found for K={k}, P={p}")
    best = json.loads(result[0])[-1]
    conn.close()
    return np.array(best["solution"])


def scope_policy_viz(policy: SCOPE,
                     obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Visualize the policy"""

    # Step 1: Normalize the screen input
    m = obs.astype(np.float32) / 255.0

    # Step 2: Apply the 2-Dimensional Discrete Cosine Transform
    dct_rows = dct(m.T, norm='ortho')
    dct_full = dct(dct_rows.T, norm='ortho')
    m_prime = dct_full[:policy.k, :policy.k].copy()

    # Step 3: Sparsification
    threshold = np.percentile(np.abs(m_prime), 25)
    m_prime_sparse = m_prime.copy()
    m_prime_sparse[np.abs(m_prime_sparse) < threshold] = 0

    # Step 4: Policy output
    action_logits = policy.weights_1 @ m_prime_sparse @ policy.weights_2 + policy.bias
    return m, m_prime, m_prime_sparse, action_logits.flatten()


def visualize_episode(game: str,
                      policy: SCOPE,
                      max_steps: int = 1000,
                      repeat_prob: float = 0.0,
                      frameskip: int = 4) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Visualize the episode"""

    # Create the environment
    env = gym.make(id=game,
                   obs_type="grayscale",
                   repeat_action_probability=repeat_prob,
                   frameskip=frameskip,
                   render_mode="rgb_array")

    # Reset the environment
    obs, _ = env.reset()

    # Initialize the frames
    frames = []

    # Run the episode
    for _ in range(max_steps):
        m, m_dct, m_sparse, logits = scope_policy_viz(policy, obs)
        action = int(np.argmax(logits))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append((m, m_dct, m_sparse, logits))
        if done:
            break
    env.close()
    return frames


def animate_frames(frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                   k: int,
                   output_size: int,
                   output_path: str = "out/scope_visualization_real.gif"):
    """Animate the frames by creating a GIF"""

    # Constants
    LOGIT_YLIM = 2500

    # Create the subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.12, wspace=0.4)

    # Grayscale input
    im0 = axs[0].imshow(np.zeros((210, 160)), cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Grayscale Input")
    axs[0].axis("off")

    # Custom colormaps with black for masked (zeroed) values
    dct_cmap = cm.seismic.copy()
    dct_cmap.set_bad(color='black')

    sparse_cmap = cm.seismic.copy()
    sparse_cmap.set_bad(color='black')

    # Full DCT heatmap
    im1 = axs[1].imshow(np.zeros((k, k)), cmap=dct_cmap)
    axs[1].set_title(f"DCT (Truncated to {k}x{k})")

    # Sparse DCT heatmap
    im2 = axs[2].imshow(np.zeros((k, k)), cmap=sparse_cmap)
    axs[2].set_title("Sparse Truncated DCT")

    # Action logits bar plot
    bar_container = axs[3].bar(np.arange(output_size), np.zeros(output_size))
    axs[3].set_ylim(-LOGIT_YLIM, LOGIT_YLIM)
    axs[3].set_title("Action Logits")

    # Fixed action labels (assumes 6 discrete actions for Atari)
    action_labels = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
    axs[3].set_xticks(np.arange(len(action_labels)))
    axs[3].set_xticklabels(action_labels, rotation=45, ha="right", fontsize=9)

    # Add colorbars
    cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cbar1.set_label("DCT Coefficient Value")

    cbar2 = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Sparse DCT Value")

    def update(frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """Update the frames for the animation"""

        m, dct_vals, sparse_vals, logits = frame

        im0.set_data(m)

        # Normalize the heatmaps with the same symmetric range
        max_coeff = np.max(np.abs(dct_vals)) or 1.0

        dct_masked = np.ma.masked_where(dct_vals == 0, dct_vals)
        im1.set_data(dct_masked)
        im1.set_clim(-max_coeff, max_coeff)

        sparse_masked = np.ma.masked_where(sparse_vals == 0, sparse_vals)
        im2.set_data(sparse_masked)
        im2.set_clim(-max_coeff, max_coeff)

        # Update the action logits bar plot
        for bar, val in zip(bar_container, logits):
            clipped_val = np.clip(val, -LOGIT_YLIM, LOGIT_YLIM)
            bar.set_height(clipped_val)
            bar.set_color("red" if val == np.max(logits) else "blue")

        # Return the updated frames
        return [im0, im1, im2] + list(bar_container)

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / 60, blit=False)

    # Save the animation
    writer = PillowWriter(fps=60)
    ani.save(output_path, writer=writer)
    print(f"Animation saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="runs_5000.db", help="Path to SQLite DB")
    parser.add_argument("--game", type=str, default="ALE/SpaceInvaders-v5", help="ALE game name (e.g., Breakout-v5)")
    parser.add_argument("-k", type=int, default=75, help="Value for K override")
    parser.add_argument("-p", type=float, default=0.25, help="Value for P override")
    parser.add_argument("--frameskip", type=int, default=4, help="Frameskip value")
    parser.add_argument("--repeat_prob", type=float, default=0.0, help="Repeat action probability")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps per episode")

    args = parser.parse_args()
    sol = get_best_solution(args.db, args.game, args.k, args.p)
    dummy_env = gym.make(args.game, obs_type="grayscale", render_mode="rgb_array")
    action_space = dummy_env.action_space.n
    dummy_env.close()

    policy = SCOPE(chromosome=sol, output_size=action_space, k=args.k, p=args.p)
    frames = visualize_episode(args.game, policy, max_steps=args.max_steps)
    animate_frames(frames, args.k, action_space)


if __name__ == "__main__":
    main()
