"""
heatmap_avg_and_max.py

Generates heatmaps to visualize the performance of different hyperparameter combinations from a SQLite database.
"""

import sqlite3
import json
import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

# Path to the data and output directories
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT_FOLDER = "out"

HEATMAPS_DIR = os.path.join(OUT_FOLDER, "heatmaps")
DB_PATH_DEFAULT = os.path.join(DATA_FOLDER, "runs_500gen.db")  # change this if needed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Generate heatmaps from SQLite database of runs.")
    parser.add_argument("--db-path", default=DB_PATH_DEFAULT, help="Name of the database file in the ../data folder (default: %(default)s)")
    parser.add_argument("--save-png", action="store_true", help="Save plots in PNG format in addition to PDF")
    parser.add_argument("--save-svg", action="store_true", help="Save plots in SVG format in addition to PDF")
    return parser.parse_args()


def load_all_runs(db_path: str) -> list[tuple[str, dict, float]]:
    """
    Connect to SQLite, fetch all rows of (game, overrides_json, best_fitness).
    Returns a list of tuples: (game_str, overrides_dict, best_fitness_float).
    """
    # Find the database file
    if not os.path.isabs(db_path):
        db_path = os.path.join(DATA_FOLDER, db_path)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT game, overrides_json, best_fitness
        FROM runs
        WHERE overrides_json IS NOT NULL
    """
    cur.execute(query)
    rows = cur.fetchall()
    results = []
    for row in rows:
        game = row["game"]
        ov_json = row["overrides_json"]
        try:
            ov_dict = json.loads(ov_json)
        except json.JSONDecodeError:
            continue

        # Only keep numeric fields
        numeric_params = {}
        for k, v in ov_dict.items():
            try:
                numeric_params[k] = float(v)
            except (ValueError, TypeError):
                # Non-numeric values are skipped
                pass

        if len(numeric_params) < 2:
            # We need at least two numeric dimensions for a 2D heatmap
            continue

        best_fitness = row["best_fitness"]
        try:
            best_f = float(best_fitness)
        except (ValueError, TypeError):
            continue

        results.append((game, numeric_params, best_f))

    conn.close()
    return results


def aggregate_by_game_and_params(all_runs: list[tuple[str, dict, float]]) -> dict[str, dict]:
    """
    all_runs is a list of (game, numeric_params, best_fitness).
    Group by game. For each game:
      - Pick exactly two parameter keys (in sorted order)
      - Build a mapping: (param1_value, param2_value) -> list of best_fitness values
      - Compute both the average and the maximum over that list, then store into 2D arrays.

    Returns:
      {
        game1: {
          'param1': <name>,
          'param2': <name>,
          'grid_x': np.array(sorted unique param1 values),
          'grid_y': np.array(sorted unique param2 values),
          'avg_fitness': 2D numpy array shape (len(grid_y), len(grid_x)),
          'max_fitness': 2D numpy array shape (len(grid_y), len(grid_x))
        },
        ...
      }
    """
    by_game = defaultdict(list)
    for game, param_dict, best_f in all_runs:
        by_game[game].append((param_dict, best_f))

    out = {}
    for game, runs in by_game.items():
        # Gather all numeric‐param keys seen under this game
        keys_seen = set()
        for param_dict, _ in runs:
            keys_seen.update(param_dict.keys())

        two_keys = sorted(keys_seen)
        if len(two_keys) < 2:
            # skip if fewer than 2 distinct numeric fields
            continue

        param1, param2 = two_keys[0], two_keys[1]

        # Build a mapping (p1, p2) -> list of fitness
        collection = defaultdict(list)
        all_x = set()
        all_y = set()

        for param_dict, best_f in runs:
            if param1 in param_dict and param2 in param_dict:
                x = param_dict[param1]
                y = param_dict[param2]
                all_x.add(x)
                all_y.add(y)
                collection[(x, y)].append(best_f)

        if not all_x or not all_y:
            # No valid pairs --> skip
            continue

        sorted_x = sorted(all_x)
        sorted_y = sorted(all_y)

        avg_matrix = np.full((len(sorted_y), len(sorted_x)), np.nan, dtype=float)
        max_matrix = np.full((len(sorted_y), len(sorted_x)), np.nan, dtype=float)

        for i, yy in enumerate(sorted_y):
            for j, xx in enumerate(sorted_x):
                fitness_list = collection.get((xx, yy), [])
                if fitness_list:
                    avg_matrix[i, j] = sum(fitness_list) / len(fitness_list)
                    max_matrix[i, j] = max(fitness_list)

        out[game] = {
            "param1": param1,
            "param2": param2,
            "grid_x": np.array(sorted_x),
            "grid_y": np.array(sorted_y),
            "avg_fitness": avg_matrix,
            "max_fitness": max_matrix,
        }

    return out


def plot_and_save_heatmaps(game: str,
                           data_dict: dict,
                           output_dir: str = ".",
                           db_name: str = "",
                           save_formats: list[str] = None):
    """
    Plot two discrete-tick heatmaps side by side for one game:
      - Left:    average of all best_fitness values at each (param1, param2)
      - Right:   maximum of all best_fitness values at each (param1, param2)

    Each axis tick is an integer index labeled by the corresponding discrete parameter value.
    Saves to: output_dir/heatmap_<db_name>_<game>.<format> for each format in save_formats
    """
    if save_formats is None:
        save_formats = ['pdf']  # Default to PDF only
        
    p1 = data_dict["param1"]
    p2 = data_dict["param2"]
    X_vals = data_dict["grid_x"]    # e.g. array([1.0, 5.0, 10.0, ...])
    Y_vals = data_dict["grid_y"]    # e.g. array([0.1, 0.5, 1.0, ...])
    Z_avg = data_dict["avg_fitness"]
    Z_max = data_dict["max_fitness"]

    if Z_avg.size == 0 or np.all(np.isnan(Z_avg)):
        print(f"[{game}] no valid fitness data to plot. Skipping.")
        return

    nx = len(X_vals)
    ny = len(Y_vals)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Add main title for the game
    fig.suptitle(game, fontsize=14, y=1.02)

    # Left subplot: AVERAGE
    im_avg = axes[0].imshow(
        np.ma.masked_invalid(Z_avg),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    fig.colorbar(im_avg, ax=axes[0])
    axes[0].set_title("Avg Best Score")
    axes[0].set_xlabel(p1)
    axes[0].set_ylabel(p2)
    axes[0].set_xticks(np.arange(nx))
    axes[0].set_xticklabels([str(x) for x in X_vals], rotation=45)
    axes[0].set_yticks(np.arange(ny))
    axes[0].set_yticklabels([str(y) for y in Y_vals])

    # Right subplot: MAX
    im_max = axes[1].imshow(
        np.ma.masked_invalid(Z_max),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="plasma",
    )
    fig.colorbar(im_max, ax=axes[1])
    axes[1].set_title("Max Best Score")
    axes[1].set_xlabel(p1)
    axes[1].set_ylabel(p2)
    axes[1].set_xticks(np.arange(nx))
    axes[1].set_xticklabels([str(x) for x in X_vals], rotation=45)
    axes[1].set_yticks(np.arange(ny))
    axes[1].set_yticklabels([str(y) for y in Y_vals])

    plt.tight_layout()

    safe_game = game.replace(" ", "_").replace("/", "_")
    base_path = os.path.join(output_dir, f"heatmap_{db_name}_{safe_game}")
    
    # Save in each requested format
    for fmt in save_formats:
        out_path = f"{base_path}.{fmt}"
        plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
        print(f"[{game}] heatmap (avg & max) saved to {out_path}")
    
    plt.close(fig)


def main():
    """Main function to generate heatmaps from run data"""
    args = parse_args()
    db_path = args.db_path
    
    # Determine which formats to save
    save_formats = ['pdf']  # Always save PDF
    if args.save_png:
        save_formats.append('png')
    if args.save_svg:
        save_formats.append('svg')
    
    # Get database name without .db extension
    db_name = os.path.splitext(os.path.basename(db_path))[0]
    
    all_runs = load_all_runs(db_path)
    if not all_runs:
        print("No valid runs found in the database.")
        return

    aggregated = aggregate_by_game_and_params(all_runs)
    if not aggregated:
        print("No games had at least two numeric override parameters. Nothing to plot.")
        return

    os.makedirs(HEATMAPS_DIR, exist_ok=True)

    for game, data_dict in aggregated.items():
        plot_and_save_heatmaps(game, data_dict, output_dir=HEATMAPS_DIR, 
                              db_name=db_name, save_formats=save_formats)

    print(f"\nDone. All heatmaps are in the `{HEATMAPS_DIR}` folder.")


if __name__ == "__main__":
    main()
