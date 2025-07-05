"""
Connects to a SQLite database (with a table `runs(game, overrides_json, best_fitness, ...)`)
and, for each distinct game, computes the best fitness value and the average fitness value
for each distinct overrides_json.

Saves the best solution for each game to a .npy file in the `out/best_solutions` directory.
"""

import os
import sqlite3
import json
import statistics
import argparse
import numpy as np
from tabulate import tabulate
import csv

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT_FOLDER = "out"

DB_PATH = os.path.join(DATA_FOLDER, "runs_5000_all.db") # default path if none provided
SOL_DIR = os.path.join(OUT_FOLDER, "best_solutions")    # where we'll write the .npy files
TABLES_DIR = os.path.join(OUT_FOLDER, "tables")           # where we'll write the CSV files


def parse_args():
    """Parse command-line arguments for sorting and database path"""
    parser = argparse.ArgumentParser(description="Aggregate best_fitness stats from one or more SQLite DBs.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sort-by-average", action="store_true", help="Sort output by highest average best_fitness")
    group.add_argument("--sort-by-overall", action="store_true", help="Sort output by highest overall best_fitness")
    parser.add_argument("--export-csv", action="store_true", help="Export results to CSV file")
    parser.add_argument("--db-path", default=DB_PATH, help="Name of the database file in the ../data folder (default: %(default)s)")
    return parser.parse_args()


def export_to_csv(results: list[dict], db_path: str):
    """Export results to a CSV file in the tables directory"""
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    # Get the database name without .db extension
    db_name = os.path.splitext(os.path.basename(db_path))[0]
    csv_path = os.path.join(TABLES_DIR, f"{db_name}_stats.csv")
    
    # Prepare headers and rows
    headers = [
        "GAME", "OVERRIDES (JSON)", "BEST_ID", "OVERALL_BEST",
        "AVG_BEST", "SIGMA", "P25", "P75", "NUM_RUNS"
    ]
    
    rows = []
    for r in results:
        rows.append([
            r["game"],
            r["overrides"],
            r["best_id"],
            r["overall_best"],
            r["avg_best"],
            r["sigma"],
            r["q1"],
            r["q3"],
            r["num_runs"]
        ])
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main(db_path: str,
         sort_avg: bool = False,
         sort_overall: bool = False,
         export_csv: bool = False):
    # Ensure the output directory exists
    os.makedirs(SOL_DIR, exist_ok=True)

    # Always access the database from the data folder
    db_path = os.path.join(DATA_FOLDER, os.path.basename(db_path))

    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Find each (game, overrides_json) pair
    cur.execute("""
        SELECT DISTINCT game, overrides_json
        FROM runs
        ORDER BY game, overrides_json
    """)
    pairs = cur.fetchall()
    if not pairs:
        print("No data found in 'runs' table.")
        conn.close()
        return

    # Initialize the results list
    results = []

    # Iterate over each (game, overrides_json) pair
    for row in pairs:
        game           = row["game"]
        overrides_json = row["overrides_json"]

        # Pretty-print the overrides dict (or fallback to raw JSON)
        try:
            ov_dict   = json.loads(overrides_json)
            overrides_pretty = json.dumps(ov_dict, sort_keys=True)
        except json.JSONDecodeError:
            overrides_pretty = overrides_json

        # Fetch every run's plot_data_json, compute its all-time max, record row id
        cur.execute("""
            SELECT id, plot_data_json
            FROM runs
            WHERE game = ? AND overrides_json = ?
        """, (game, overrides_json))

        # Initialize the run_maxes list
        run_maxes = []
        for r2 in cur.fetchall():
            raw = r2["plot_data_json"]
            try:
                pd = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Each pd is list of [gen, best, avg]; take max of best
            run_max = max(float(triple[1]) for triple in pd)
            run_maxes.append((r2["id"], run_max))

        if not run_maxes:
            continue

        # Extract stats
        best_values     = [v for (_id, v) in run_maxes]
        overall_best    = max(best_values)
        overall_best_id = next(_id for (_id, v) in run_maxes if v == overall_best)
        avg_best        = statistics.mean(best_values)
        sigma           = statistics.pstdev(best_values)
        q1, _, q3       = statistics.quantiles(best_values, n=4)
        num_runs        = len(best_values)

        # Fetch that row's best_individuals_json and save the best solution
        cur.execute("""
            SELECT best_individuals_json
            FROM runs
            WHERE id = ?
        """, (overall_best_id,))

        # Fetch the best_individuals_json
        bi_row = cur.fetchone()

        # If the best_individuals_json is valid, save the best solution
        if bi_row and bi_row["best_individuals_json"]:
            try:
                individuals = json.loads(bi_row["best_individuals_json"])
            except json.JSONDecodeError:
                individuals = []
            if individuals:
                # pick the entry with the highest fitness
                best_entry = max(individuals, key=lambda e: float(e.get("fitness", -float("inf"))))
                sol = best_entry.get("solution")
                if sol is not None:
                    arr = np.array(sol)
                    # Build a filename containing game, override-hash, and the row id
                    game_safe = game.replace("/", "_")
                    fname     = f"{game_safe}_K{ov_dict['K']}_P{ov_dict['P']}_individual_{best_entry['fitness']:.2f}.npy"
                    # Create subdirectory based on db name
                    db_name = os.path.splitext(os.path.basename(db_path))[0]
                    sub_dir = os.path.join(SOL_DIR, db_name)
                    os.makedirs(sub_dir, exist_ok=True)
                    out_path  = os.path.join(sub_dir, fname)
                    np.save(out_path, arr)

        # Collect for tabulation
        results.append({
            "game": game,
            "overrides": overrides_pretty,
            "best_id": overall_best_id,
            "overall_best": overall_best,
            "avg_best": avg_best,
            "sigma": sigma,
            "q1": q1,
            "q3": q3,
            "num_runs": num_runs
        })

    conn.close()

    # Optional sorting
    if sort_avg:
        results.sort(key=lambda r: r["avg_best"], reverse=True)
    elif sort_overall:
        results.sort(key=lambda r: r["overall_best"], reverse=True)

    # Print table
    headers = [
        "GAME", "OVERRIDES (JSON)", "BEST_ID", "OVERALL_BEST",
        "AVG_BEST ± σ", "P25", "P75", "NUM_RUNS"
    ]
    rows = [
        [
            r["game"],
            r["overrides"],
            r["best_id"],
            f"{r['overall_best']:.4f}",
            f"{r['avg_best']:.4f} ± {r['sigma']:.4f}",
            f"{r['q1']:.4f}",
            f"{r['q3']:.4f}",
            r["num_runs"]
        ]
        for r in results
    ]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", stralign="left"))

    # Export to CSV if requested
    if export_csv:
        export_to_csv(results, db_path)


if __name__ == "__main__":
    args = parse_args()
    main(
        db_path=args.db_path,
        sort_avg=args.sort_by_average,
        sort_overall=args.sort_by_overall,
        export_csv=args.export_csv
    )
