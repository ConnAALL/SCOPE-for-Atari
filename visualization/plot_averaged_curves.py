"""
Generate averaged learning-curve and summary plots from two SQLite databases
"""

import os
import json
import sqlite3
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scienceplots

plt.style.use(['science', 'no-latex'])

# Directory that holds the SQLite DBs (relative to the project root)
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Directory for output files (relative to this script)
OUT_FOLDER = os.path.join(os.path.dirname(__file__), "out")

# Default paths (override via CLI)
PRIMARY_DB   = "runs_5000_all.db"
SECONDARY_DB = "runs_500gen.db"
OUTPUT_DIR   = "plots_5000"
OUT_EXT      = ".pdf"

# Color mapping for parameter combinations
PARAM_COLORS = {
    "K=50, P=0.95": "#1f77b4",   # blue
    "K=75, P=0.25": "#ff7f0e",   # orange
    "K=125, P=10": "#2ca02c",    # green
    "K=125, P=25": "#d62728",    # red
    "K=150, P=0.9": "#9467bd",   # purple
    "K=150, P=10": "#8c564b",    # brown
}


def ensure_output_dir(path: str):
    """Ensure the output directory exists"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def sanitize_game_name(game: str) -> str:
    """Sanitize the game name for file naming"""
    return game.replace("/", "_")


def parse_overrides(overrides_json: str) -> tuple[int, float]:
    """Parse the overrides JSON string into K and P values"""
    try:
        d = json.loads(overrides_json)
        return d.get("K", None), d.get("P", None)
    except json.JSONDecodeError:
        return None, None


def load_distinct_games_and_overrides(cursor: sqlite3.Cursor) -> dict[str, list[str]]:
    """
    From PRIMARY_DB: returns dict mapping game → list of override JSON strings.
    """
    cursor.execute("""
        SELECT DISTINCT game, overrides_json
        FROM runs
        ORDER BY game, overrides_json
    """)
    mapping = {}
    for row in cursor.fetchall():
        ov = row["overrides_json"]
        if ov != '{"K": 142, "P": 22}':  # your original filter
            mapping.setdefault(row["game"], []).append(ov)
    return mapping


def load_overrides_for_game(cursor: sqlite3.Cursor,
                            game: str) -> list[str]:
    """
    From any DB cursor: returns list of override JSON strings for the given game.
    """
    cursor.execute("""
        SELECT DISTINCT overrides_json
        FROM runs
        WHERE game = ?
        ORDER BY overrides_json
    """, (game,))
    return [row["overrides_json"] for row in cursor.fetchall()]


def fetch_plot_data(cursor: sqlite3.Cursor,
                    game: str,
                    overrides_json: str) -> list[list[tuple[int, float, float]]]:
    """
    Returns list of runs; each run is a list of [gen, best, avg].
    """
    cursor.execute("""
        SELECT plot_data_json
        FROM runs
        WHERE game = ? AND overrides_json = ?
    """, (game, overrides_json))
    runs = []
    for r in cursor.fetchall():
        try:
            runs.append(json.loads(r["plot_data_json"]))
        except json.JSONDecodeError:
            continue
    return runs


def average_best_per_generation(all_plot_data: list[list[tuple[int, float, float]]]) -> tuple[list[int], np.ndarray]:
    """
    Given list of runs x generations, returns:
      evals: [gen*20, ...]
      mean_best: array of shape (G,)
    """
    if not all_plot_data:
        return [], np.array([])
    G = len(all_plot_data[0])
    M = len(all_plot_data)
    mat = np.zeros((M, G), dtype=float)
    for i, run in enumerate(all_plot_data):
        if len(run) != G:
            raise ValueError("Inconsistent generation counts!")
        for j, triple in enumerate(run):
            mat[i, j] = float(triple[1])
    mean_best = np.mean(mat, axis=0)
    evals = [int(all_plot_data[0][i][0]) * 1 for i in range(G)] #Modify X axis
    return evals, mean_best


def make_individual_plot(game: str,
                         ov: str,
                         evals: list[int],
                         mean_best: np.ndarray,
                         runs: list[list[tuple[int, float, float]]],
                         output_dir: str):
    """Make an individual plot for a given game, override, and data"""

    # Sanitize the game name and parse the overrides
    game_safe = sanitize_game_name(game)
    k, p = parse_overrides(ov)
    param_key = f"K={k}, P={p}" if k is not None else ov
    color = PARAM_COLORS.get(param_key, 'black')
    fname = (f"{game_safe}_K{k}_P{p}_avg_curve{OUT_EXT}"
             if k is not None
             else f"{game_safe}_OVERRIDES_{abs(hash(ov))}_avg_curve{OUT_EXT}")

    # Calculate the percentile bands and champion
    M, G = len(runs), len(evals)
    mat = np.zeros((M, G), dtype=float)
    for i, run in enumerate(runs):
        for j, t in enumerate(run):
            mat[i, j] = float(t[1])

    p1, p10 = np.percentile(mat, 99, axis=0), np.percentile(mat, 90, axis=0)
    p25, p75 = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
    p5 = np.percentile(mat, 95, axis=0)
    champion = np.maximum.accumulate(np.max(mat, axis=0))

    # Make the plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(evals, p1, p10, alpha=0.3, color=color)
    plt.fill_between(evals, p25, p75, alpha=0.3, color=color)
    plt.plot(evals, p5, '--', linewidth=1, label='95th pct', color=color)
    plt.plot(evals, champion, ':', linewidth=1, label='Champion', color=color)
    plt.plot(evals, mean_best, linewidth=1, label='Mean Best', color=color)
    plt.xlabel("Evaluation Count")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.ylim(0, 4250)
    plt.tight_layout()

    # Save the plot
    out = os.path.join(output_dir, fname)
    plt.savefig(out, format='pdf')
    plt.close()
    print(f"  • Saved individual plot ({param_key}) → {out}")


def make_combined_plot(game: str,
                       override_list: list[str],
                       data_per_override: dict[str, tuple[list[int], np.ndarray]],
                       output_dir: str):
    """Make a combined plot for a given game and overrides"""

    # Sanitize the game name
    game_safe = sanitize_game_name(game)
    plt.figure(figsize=(12, 6))
    markers = ['o','s','D','^','v','<','>','p','*','h','H','X','d']
    n = len(override_list)

    # Make the plot
    for idx, ov in enumerate(override_list):
        evals, mean_best = data_per_override[ov]
        k, p = parse_overrides(ov)
        param_key = f"K={k}, P={p}"
        color = PARAM_COLORS.get(param_key, 'black')
        
        plt.plot(evals, mean_best,
                 color=color, linestyle='-', linewidth=1.5, alpha=0.3)
        mk = markers[idx % len(markers)]
        interval = max(1, len(evals)//10)
        offset = int(interval/(n+1)*(idx+1)) % interval
        plt.plot(evals, mean_best,
                 linestyle='None', marker=mk, markersize=6,
                 markerfacecolor='none', markeredgewidth=1.0,
                 markevery=(offset, interval),
                 color=color, alpha=1.0,
                 label=param_key)

    # Add labels and save the plot
    plt.xlabel("Evaluation Count")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend(loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()

    out = os.path.join(output_dir, f"{game_safe}_combined{OUT_EXT}")
    plt.savefig(out, format='pdf')
    plt.close()
    print(f"  • Saved combined plot → {out}")


def make_bar_chart(game: str,
                   avg_fb_sec: dict[str, float],
                   finals_sec: dict[str, list[float]],
                   output_dir: str):
    """Make a standalone bar chart for final scores from secondary database"""
    
    # Sanitize the game name
    game_safe = sanitize_game_name(game)
    
    # Sort overrides by K and P values
    ov2s = sorted(avg_fb_sec.keys(),
                  key=lambda o: (parse_overrides(o)[0] or float('inf'),
                                 parse_overrides(o)[1] or float('inf')))
    
    # Calculate statistics
    means = np.array([avg_fb_sec[o] for o in ov2s])
    q1s   = np.array([np.percentile(finals_sec[o], 25) for o in ov2s])
    q3s   = np.array([np.percentile(finals_sec[o], 75) for o in ov2s])
    mins  = np.array([np.min(finals_sec[o]) for o in ov2s])
    maxs  = np.array([np.max(finals_sec[o]) for o in ov2s])
    lower = np.clip(means - q1s, 0, None)
    upper = np.clip(q3s - means, 0, None)
    yerr  = [lower, upper]
    x = np.arange(len(ov2s))

    # Get colors for each parameter combination
    labels = [f"K={parse_overrides(o)[0]}, P={parse_overrides(o)[1]}" for o in ov2s]
    colors = [PARAM_COLORS.get(label, 'grey') for label in labels]
    
    to_fill = set(PARAM_COLORS.keys())
    fill_idx = [i for i,lab in enumerate(labels) if lab in to_fill]
    hollow_idx = [i for i in range(len(labels)) if i not in fill_idx]

    # Create the figure
    plt.figure(figsize=(12, 6))
    
    # Plot bars + error bars with specific colors
    bars = plt.bar(x, means, yerr=yerr, capsize=4, alpha=0.5, color=colors, edgecolor='k', linewidth=1)

    # Hollow markers for everything
    plt.scatter(x[hollow_idx], maxs[hollow_idx],
                marker='^', facecolors='none', alpha=0.7, edgecolors='black', s=60, label='Max')
    plt.scatter(x[hollow_idx], mins[hollow_idx],
                marker='v', facecolors='none', alpha=0.7, edgecolors='black', s=60, label='Min')

    # Filled markers with specific colors
    for idx in fill_idx:
        color = colors[idx]
        plt.scatter([x[idx]], [maxs[idx]],
                    marker='^', facecolors=color, alpha=0.7, edgecolors='black', s=60)
        plt.scatter([x[idx]], [mins[idx]],
                    marker='v', facecolors=color, alpha=0.7, edgecolors='black', s=60)

    # Add labels and formatting
    plt.xticks(x)
    plt.gca().set_xticklabels(labels, rotation=90, ha='center', fontsize='small')
    plt.margins(x=0.01)
    plt.title(f"{game} - Parameter Sweep Final Scores")
    plt.xlabel("Parameters")
    plt.ylabel("Average Final Score")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()

    # Save the plot
    out = os.path.join(output_dir, f"{game_safe}_bar_chart{OUT_EXT}")
    plt.savefig(out, format='pdf')
    plt.close()
    print(f"  • Saved bar chart → {out}")


def make_multiplot_with_final(game: str,
                              primary_ovs: list[str],
                              data_p: dict[str, tuple[list[int], np.ndarray]],
                              runs_p: dict[str, list[list[tuple[int, float, float]]]],
                              avg_fb_sec: dict[str, float],
                              finals_sec: dict[str, list[float]],
                              output_dir: str):
    """
    Top:   combined mean-best curves
    Mid:   2 rows x 3 cols of individual plots
    Bot:   avg-final-best bar chart (from secondary DB)
    """
    IND_ROWS, IND_COLS = 2, 3
    game_safe = sanitize_game_name(game)

    # Calculate the number of bars and the width of the plot
    num_bars = len(avg_fb_sec)
    bar_width_inches = num_bars * 0.15
    base_width = IND_COLS * 4
    total_width  = max(base_width, bar_width_inches)
    total_height = 4 + IND_ROWS * 3 + 4

    # Create the figure
    fig = plt.figure(figsize=(total_width, total_height))
    total_rows = 1 + IND_ROWS + 1
    gs = GridSpec(nrows=total_rows,
                  ncols=IND_COLS,
                  figure=fig,
                  height_ratios=[3] + [2]*IND_ROWS + [3],
                  hspace=0.3, wspace=0.4)

    # Combined plot on top
    ax0 = fig.add_subplot(gs[0, :])
    markers = ['o','s','D','^','v','<','>','p','*','h','H','X','d']
    n = len(primary_ovs)
    for idx, ov in enumerate(primary_ovs):
        evals, mb = data_p[ov]
        k, p = parse_overrides(ov)
        param_key = f"K={k}, P={p}"
        color = PARAM_COLORS.get(param_key, 'black')
        
        ax0.plot(evals, mb,
                 color=color, linestyle='-', linewidth=1.5, alpha=0.3)
        mk = markers[idx % len(markers)]
        interval = max(1, len(evals)//10)
        offset = int(interval/(n+1)*(idx+1)) % interval
        ax0.plot(evals, mb,
                 linestyle='None', marker=mk, markersize=6,
                 markerfacecolor='none', markeredgewidth=1.0,
                 markevery=(offset, interval),
                 color=color, alpha=1.0,
                 label=param_key)
    ax0.set_title("High Evaluation Count Trials")
    ax0.set_xlabel("Generation")
    ax0.set_ylabel("Score")
    ax0.grid(True)
    ax0.legend(loc='upper left', fontsize='small', ncol=2)

    # Individual 2x3 grid in the middle
    max_plots = IND_ROWS * IND_COLS
    for idx, ov in enumerate(primary_ovs[:max_plots]):
        r = 1 + idx // IND_COLS
        c = idx % IND_COLS
        ax = fig.add_subplot(gs[r, c])

        runs = runs_p[ov]
        M, G = len(runs), len(runs[0])
        mat = np.zeros((M, G), dtype=float)
        for i, run in enumerate(runs):
            for j, t in enumerate(run):
                mat[i, j] = float(t[1])

        p1, p10 = np.percentile(mat, 99, axis=0), np.percentile(mat, 90, axis=0)
        p25, p75 = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
        p5 = np.percentile(mat, 95, axis=0)
        champ = np.maximum.accumulate(np.max(mat, axis=0))
        evals, mb = data_p[ov]
        
        k, p = parse_overrides(ov)
        param_key = f"K={k}, P={p}"
        color = PARAM_COLORS.get(param_key, 'black')

        ax.fill_between(evals, p1, p10, alpha=0.3, color=color)
        ax.fill_between(evals, p25, p75, alpha=0.3, color=color)
        ax.plot(evals, champ, ':', linewidth=1, label='Champ', color=color)
        ax.plot(evals, p5, '-', alpha=0.5, linewidth=1, label='95th', color=color)
        ax.plot(evals, mb, linewidth=1, label='Mean', color=color)
        ax.legend(loc='upper left', fontsize='small', ncol=1)
        ax.set_title(param_key, fontsize='small')
        ax.grid(True)
        ax.set_ylim(0, 4250)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Score")

    # Blank out unused slots
    for e in range(len(primary_ovs), max_plots):
        r, c = 1 + e // IND_COLS, e % IND_COLS
        fig.add_subplot(gs[r, c]).axis('off')

    # Final-best bar chart on the bottom
    axf = fig.add_subplot(gs[-1, :])
    ov2s = sorted(avg_fb_sec.keys(),
                  key=lambda o: (parse_overrides(o)[0] or float('inf'),
                                 parse_overrides(o)[1] or float('inf')))
    means = np.array([avg_fb_sec[o] for o in ov2s])
    q1s   = np.array([np.percentile(finals_sec[o], 25) for o in ov2s])
    q3s   = np.array([np.percentile(finals_sec[o], 75) for o in ov2s])
    mins  = np.array([np.min(finals_sec[o]) for o in ov2s])
    maxs  = np.array([np.max(finals_sec[o]) for o in ov2s])
    lower = np.clip(means - q1s, 0, None)
    upper = np.clip(q3s - means, 0, None)
    yerr  = [lower, upper]
    x = np.arange(len(ov2s))

    # Get colors for each parameter combination
    labels = [f"K={parse_overrides(o)[0]}, P={parse_overrides(o)[1]}" for o in ov2s]
    colors = [PARAM_COLORS.get(label, 'grey') for label in labels]
    
    to_fill = set(PARAM_COLORS.keys())
    fill_idx = [i for i,lab in enumerate(labels) if lab in to_fill]
    hollow_idx = [i for i in range(len(labels)) if i not in fill_idx]

    # Plot bars + error‐bars with specific colors
    bars = axf.bar(x, means, yerr=yerr, capsize=4, alpha=0.5, color=colors, edgecolor='k', linewidth=1)

    # Hollow markers for everything
    axf.scatter(x[hollow_idx], maxs[hollow_idx],
                marker='^', facecolors='none',alpha=0.7, edgecolors='black', s=60, label='Max')
    axf.scatter(x[hollow_idx], mins[hollow_idx],
                marker='v', facecolors='none',alpha=0.7, edgecolors='black', s=60, label='Min')

    # Filled markers with specific colors
    for idx in fill_idx:
        color = colors[idx]
        axf.scatter([x[idx]], [maxs[idx]],
                    marker='^', facecolors=color,alpha=0.7, edgecolors='black', s=60)
        axf.scatter([x[idx]], [mins[idx]],
                    marker='v', facecolors=color, alpha=0.7,edgecolors='black', s=60)

    # Add a legend entry for selected parameters
    axf.set_xticks(x)
    axf.set_xticklabels(labels, rotation=90, ha='center', fontsize='x-small')
    axf.margins(x=0.01)
    axf.set_title("Low Evaluation Count Parameter Sweep")
    axf.set_xlabel("Parameters")
    axf.set_ylabel("Average Final Score")
    axf.grid(axis='y', linestyle='--', alpha=0.5)
    axf.legend(loc='upper left', fontsize='small')

    out = os.path.join(output_dir, f"{game_safe}_multiplot_full{OUT_EXT}")
    fig.savefig(out, format='pdf')
    plt.close(fig)
    print(f"  • Saved full multiplot → {out}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate averaged learning-curve plots from two SQLite databases.")
    parser.add_argument("--primary-db", default=PRIMARY_DB, help="Primary database filename located in the ./data folder (default: %(default)s)")
    parser.add_argument("--secondary-db", default=SECONDARY_DB, help="Secondary database filename located in the ./data folder (default: %(default)s)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory where the generated plots will be written (default: %(default)s)")
    return parser.parse_args()


def main(primary_db: str,
         secondary_db: str,
         output_dir: str):
    """Main function to generate the plots"""

    # Resolve relative database paths against the local data folder
    if not os.path.isabs(primary_db):
        primary_db = os.path.join(DATA_FOLDER, primary_db)
    if not os.path.isabs(secondary_db):
        secondary_db = os.path.join(DATA_FOLDER, secondary_db)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(OUT_FOLDER, output_dir)

    ensure_output_dir(output_dir)

    # Connect to the databases
    conn1 = sqlite3.connect(primary_db, timeout=30.0)
    conn1.row_factory = sqlite3.Row
    cur1  = conn1.cursor()

    conn2 = sqlite3.connect(secondary_db, timeout=30.0)
    conn2.row_factory = sqlite3.Row
    cur2  = conn2.cursor()

    # Load the distinct games and overrides
    game_to_ovs = load_distinct_games_and_overrides(cur1)
    if not game_to_ovs:
        print("No data found in primary DB.")
        return

    # Process each game
    for game, primary_ovs in game_to_ovs.items():
        print(f"\nProcessing game: {game}")

        # Initialize the data
        data_p = {}
        runs_p = {}

        # Fetch the data for each override
        for ov in primary_ovs:
            runs = fetch_plot_data(cur1, game, ov)
            if not runs:
                print(f"  ➔ [primary] no data for {ov}")
                continue
            runs_p[ov] = runs
            evals, mb = average_best_per_generation(runs)
            data_p[ov] = (evals, mb)
            make_individual_plot(game, ov, evals, mb, runs, output_dir)

        # Fetch the data for the secondary overrides
        secondary_ovs = load_overrides_for_game(cur2, game)
        avg_fb_sec    = {}
        finals_sec    = {}
        for ov2 in secondary_ovs:
            runs2 = fetch_plot_data(cur2, game, ov2)
            if not runs2:
                print(f"  ➔ [secondary] no data for {ov2}")
                continue
            finals = [float(r[-1][1]) for r in runs2]
            finals_sec[ov2] = finals
            avg_fb_sec[ov2] = np.mean(finals)

        # Make the multiplot with the final
        if data_p and avg_fb_sec:
            make_multiplot_with_final(game,
                                      list(data_p.keys()),
                                      data_p,
                                      runs_p,
                                      avg_fb_sec,
                                      finals_sec,
                                      output_dir)

            # Make the combined plot
            make_combined_plot(game, list(data_p.keys()), data_p, output_dir)

            # Make the bar chart
            make_bar_chart(game, avg_fb_sec, finals_sec, output_dir)

    # Close the databases
    conn1.close()
    conn2.close()
    print("\nAll SVG plots generated.")


if __name__ == "__main__":
    _args = parse_args()
    main(primary_db=_args.primary_db,
         secondary_db=_args.secondary_db,
         output_dir=_args.output_dir)
