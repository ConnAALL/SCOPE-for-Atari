import os
import sys
import sqlite3
import json
import argparse
import itertools

import yaml
import numpy as np
import cma
import gymnasium as gym
import ale_py
import ray

from SCOPE import *


# Loading and saving the base config
with open("config.yaml", "r") as f: base_config = yaml.safe_load(f)

DEFAULTS = {
    # Environment Parameters
    "ENV_NAME": base_config["ENV_NAME"],
    "REPEAT_ACTION_PROBABILITY": base_config["REPEAT_ACTION_PROBABILITY"],
    "FRAMESKIP": base_config["FRAMESKIP"],
    "OBS_TYPE": base_config["OBS_TYPE"],
    "EPISODES_PER_INDIVIDUAL": base_config["EPISODES_PER_INDIVIDUAL"],
    "MAX_STEPS_PER_EPISODE": base_config["MAX_STEPS_PER_EPISODE"],

    # CMA-ES Parameters
    "CMA_SIGMA": base_config["CMA_SIGMA"],
    "POPULATION_SIZE": base_config["POPULATION_SIZE"],
    "GENERATIONS": base_config["GENERATIONS"],

    # Logging Configuration
    "VERBOSITY_LEVEL": base_config["VERBOSITY_LEVEL"],

    # SCOPE Parameters
    "K": base_config["K"],
    "P": base_config["p"],
}

# Arguments for the parameter sweep mode
SWEEP_K = [10, 25, 50, 75, 100, 125, 150]
SWEEP_P = [0, 10, 25, 50, 75, 90, 95]

# Arguements for the partial sweep mode
PARTIAL_SWEEP_PARAMETERS = [[150, 10], [125, 25], [125, 10], [142, 22]]

# Arguements for the multi-game sweep mode
MULTI_GAME_SWEEP_GAME_CONFIGS = [
    {"ENV_NAME": "ALE/SpaceInvaders-v5", "K": 150, "P": 10},
    {"ENV_NAME": "ALE/Breakout-v5", "K": 125, "P": 25},
    {"ENV_NAME": "ALE/Pong-v5", "K": 125, "P": 10},
    {"ENV_NAME": "ALE/Asteroids-v5", "K": 142, "P": 22},
]

# Repeats per combo
REPEATS_PER_COMBO = 100



def make_silent_env(game_name: str,
                    repeat_action_prob: float,
                    frameskip: int) -> gym.Env:
    """Create a silent Gym environment without the default console output of ALE"""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    os.dup2(devnull_fd, stdout_fd)
    os.dup2(devnull_fd, stderr_fd)
    try:
        env = gym.make(id=game_name,
                       obs_type="grayscale",
                       repeat_action_probability=repeat_action_prob,
                       frameskip=frameskip)
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)
    return env


def _evaluate_individual(args: tuple[np.ndarray, int, str, int, int, int, float, int, int, float, int]) -> tuple[int, float, list[list[int | float]]]:
    """
    Evaluate an individual policy. Returns
      (individual_index, average_reward, rows_list).
    where rows_list is a list of [gen, indv, ep, ep_reward].
    """
    # Unpack the arguments
    (
        solution,
        individual,
        game,
        gen,
        output_size,
        K,
        P,
        EPISODES_PER_INDIVIDUAL,
        MAX_STEPS_PER_EPISODE,
        REPEAT_ACTION_PROBABILITY,
        FRAMESKIP,
    ) = args

    # Create the policy
    policy = SCOPE(chromosome=solution,
                   output_size=output_size,
                   k=K,
                   p=P)

    # Create the environment
    env = make_silent_env(game_name=game,
                          repeat_action_prob=REPEAT_ACTION_PROBABILITY,
                          frameskip=FRAMESKIP)

    # Run the episodes
    total_reward = 0.0
    rows = []  # each row: [generation, individual_idx, episode_idx, ep_reward]

    for episode in range(EPISODES_PER_INDIVIDUAL):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            state = obs.astype(np.float32) / 255.0
            prefs = policy.forward(state)
            action = int(np.argmax(prefs))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        total_reward += ep_reward
        rows.append([gen + 1, individual + 1, episode + 1, ep_reward])

    env.close()
    avg_reward = total_reward / EPISODES_PER_INDIVIDUAL
    return individual, avg_reward, rows


@ray.remote(num_cpus=1)
def run_benchmark_with_overrides(overrides: dict,
                                 run_id: int) -> dict:
    """
    Merge 'overrides' onto DEFAULTS, run CMA-ES + Gym benchmark for every game in ENV_NAMES,
    collect everything in-memory and return a dict:
    ```
       {
         "run_id": <int>,
         "overrides": {…},
         "games": {
           "<ENV_NAME>": {
             "fitness_log": [ [gen, indv, ep, ep_reward], … ],
             "best_individuals": [
               {
                 "generation": g,
                 "individual_index": i,
                 "fitness": f,
                 "solution": [ … ]  # Python list, not numpy array
               },
               …
             ],
             "plot_data": [ [gen, best_fitness, avg_fitness], … ]
           },
           …
         }
       }
    ```
    """
    # Build the effective config
    config = DEFAULTS.copy()
    config.update(overrides)

    # Unpack the config
    ENV_NAME = config["ENV_NAME"]
    EPISODES_PER_INDIVIDUAL = config["EPISODES_PER_INDIVIDUAL"]
    MAX_STEPS_PER_EPISODE = config["MAX_STEPS_PER_EPISODE"]
    POPULATION_SIZE = config["POPULATION_SIZE"]
    CMA_SIGMA = config["CMA_SIGMA"]
    GENERATIONS = config["GENERATIONS"]
    VERBOSITY_LEVEL = config["VERBOSITY_LEVEL"]
    K = config["K"]
    P = config["P"]
    REPEAT_ACTION_PROBABILITY = config["REPEAT_ACTION_PROBABILITY"]
    FRAMESKIP = config["FRAMESKIP"]

    # Initialize the run data
    run_data = {
        "run_id": run_id,
        "overrides": overrides,
        "games": {}  # will fill in per‐environment below
    }

    if VERBOSITY_LEVEL >= 1:
        print(f"[Run {run_id}] Starting benchmark for {ENV_NAME} with overrides {overrides}")

    # In‐memory containers for this game
    fitness_log = []         # rows of [gen, indv_idx, ep_idx, ep_reward]
    best_individuals = []    # list of {generation, individual_index, fitness, solution_list}
    plot_data = []           # rows of [generation, best_fitness, avg_fitness]

    # Create a silent env just to get action_space.n
    env = make_silent_env(ENV_NAME, REPEAT_ACTION_PROBABILITY, FRAMESKIP)
    output_size = env.action_space.n
    chromosome_size = compute_chromosome_size(K, output_size)
    env.close()

    # Initialize the CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(
        np.zeros(chromosome_size),
        CMA_SIGMA,
        {"popsize": POPULATION_SIZE}
    )

    # Initialize the best individual fitness and solution
    best_indiv_fitness_so_far = float("-inf")
    best_indiv_solution_so_far = None

    # Run the generations
    for gen in range(GENERATIONS):
        # Ask the optimizer for new solutions
        solutions = es.ask()
        args_list = [
            (
                solutions[i],
                i,
                ENV_NAME,
                gen,
                output_size,
                K,
                P,
                EPISODES_PER_INDIVIDUAL,
                MAX_STEPS_PER_EPISODE,
                VERBOSITY_LEVEL,
                REPEAT_ACTION_PROBABILITY,
                FRAMESKIP,
            )
            for i in range(len(solutions))
        ]

        # Evaluate each individual serially. (One CPU per Ray task.)
        results = [ _evaluate_individual(args) for args in args_list ]

        # Build fitness for CMA‐ES, and collect per‐episode rows
        fitness_for_cma = []
        for indiv_idx, avg_reward, rows in results:
            # Append every row to fitness_log
            fitness_log.extend(rows)

            # CMA‐ES wants "cost" = –(average reward)
            fitness_for_cma.append((indiv_idx, -avg_reward))

            if VERBOSITY_LEVEL >= 2:
                for row in rows:
                    g, indv_i, ep_i, rew = row
                    print(
                        f"[Run {run_id}][{ENV_NAME}][GEN: {g}/{GENERATIONS}]"
                        f"[INDV: {indv_i}/{POPULATION_SIZE}]"
                        f"[EPS: {ep_i}/{EPISODES_PER_INDIVIDUAL}]"
                        f"[REW: {rew:.2f}]"
                    )

        # Sort by individual index so CMA‐ES lines up correctly
        # It should be sorted by individual index already, but just in case
        fitness_for_cma.sort(key=lambda x: x[0])
        costs = [ f for _, f in fitness_for_cma ]
        es.tell(solutions, costs)

        # Compute this generation's best & average (in reward‐space)
        best_idx = int(np.argmin(costs))
        best_val = -costs[best_idx]
        avg_val = -sum(costs) / len(costs)

        # If it's a new global best across all generations so far:
        if best_val > best_indiv_fitness_so_far:
            best_indiv_fitness_so_far = best_val
            best_indiv_solution_so_far = solutions[best_idx].copy()

            # Record that "global best snapshot" as a Python dict
            best_individuals.append({
                "generation": gen + 1,
                "individual_index": best_idx + 1,
                "fitness": best_val,
                # Convert NumPy array to Python list so that json.dumps() works
                "solution": best_indiv_solution_so_far.tolist()
            })

            if VERBOSITY_LEVEL >= 1:
                print(f"[Run {run_id}][{ENV_NAME}][GEN {gen+1}] NEW GLOBAL BEST: {best_val:.2f}")

        if VERBOSITY_LEVEL >= 1:
            print(f"[Run {run_id}][{ENV_NAME}][GEN {gen+1}] BEST: {best_val:.2f}  AVG: {avg_val:.2f}")

        # Always append this generation's (best, avg) to plot_data
        plot_data.append([ gen + 1, best_val, avg_val ])

    # End of all generations for this game
    run_data["games"][ENV_NAME] = {
        "fitness_log": fitness_log,
        "best_individuals": best_individuals,
        "plot_data": plot_data,
    }

    if VERBOSITY_LEVEL >= 1:
        print(f"[Run {run_id}] Finished benchmark for {ENV_NAME}\n")

    return run_data


def _ensure_db_and_table(db_path: str) -> sqlite3.Connection:
    """
    Create (or open) the SQLite database file at db_path, and ensure the 'runs' table exists.
    Returns a sqlite3.Connection object (with row-based autocommit turned on).
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL;")     # Allow concurrent reads/writes
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS runs (
        id                      INTEGER   PRIMARY KEY AUTOINCREMENT,
        run_id                  INTEGER   NOT NULL,
        game                    TEXT      NOT NULL,
        overrides_json          TEXT      NOT NULL,
        fitness_log_json        TEXT      NOT NULL,
        best_individuals_json   TEXT      NOT NULL,
        plot_data_json          TEXT      NOT NULL,
        total_generations       INTEGER   NOT NULL,
        best_fitness            REAL      NOT NULL,
        inserted_at             DATETIME  DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn.execute(create_table_sql)
    conn.commit()
    return conn


def main():
    """
    ENTRY POINT:
    ------------
    1) Parse command line arguments
    2) Initialize Ray
    3) Create/open SQLite database + ensure table
    4) Launch Ray tasks for the entire sweep
    5) As each task finishes, immediately serialize + INSERT into 'runs'
    6) At the end, close DB & shutdown Ray
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parameter sweep for SCOPE on Atari games')
    parser.add_argument('--mode', 
                        choices=['full-sweep', 'partial-sweep', 'multi-game-sweep'],
                        required=True,
                        help='Mode of operation: full-sweep for all combinations, partial-sweep for specific combinations, multi-game-sweep for optimal combinations across multiple games')
    args = parser.parse_args()

    print(f"Running in {args.mode} mode")

    # Start the Ray cluster
    project_root = os.path.dirname(os.path.abspath(__file__))
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env={
            # Ray will upload everything under project_root to the remote workers
            "working_dir": project_root,
            # Exclude large local database files from being shipped to remote workers
            "excludes": [
                '../data/runs.db',
                '../data/runs_5000.db',
                '../data/runs_0_100.db',
                '../data/runs_0_100_5000.db',
                '../data/runs_5000_gen.db',
                '../data/runs_500gen.db',
                'heatmaps', 'plots', 'plots_5000', 'plots_0_100'
            ]
        }
    )

    # Open or create the SQLite file inside the project's data/ directory
    data_dir = os.path.abspath(os.path.join(project_root, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)

    # Use different database names based on mode
    if args.mode == 'full-sweep':
        DB_PATH = os.path.join(data_dir, "runs_full_sweep.db")
    elif args.mode == 'partial-sweep':
        DB_PATH = os.path.join(data_dir, "runs_partial_sweep.db")
    elif args.mode == 'multi-game-sweep':
        DB_PATH = os.path.join(data_dir, "runs_multi_game_sweep.db")
    conn = _ensure_db_and_table(DB_PATH)
    cursor = conn.cursor()

    # Generate parameter combinations based on mode
    all_overrides = []
    run_id_counter = 0
    
    if args.mode == 'full-sweep':
        # Full sweep: all combinations of K and P values
        print(f"Full sweep: {len(SWEEP_K)} K values x {len(SWEEP_P)} P values x {REPEATS_PER_COMBO} repeats = {len(SWEEP_K) * len(SWEEP_P) * REPEATS_PER_COMBO} total runs")
        
        for _ in range(REPEATS_PER_COMBO):
            for (k_val, p_val) in itertools.product(SWEEP_K, SWEEP_P):
                run_id_counter += 1
                overrides = { "K": k_val, "P": p_val }
                all_overrides.append((overrides, run_id_counter))
    
    elif args.mode == 'partial-sweep':
        # Partial sweep: specific K/P combinations
        print(f"Partial sweep: {len(PARTIAL_SWEEP_PARAMETERS)} parameter combinations x {REPEATS_PER_COMBO} repeats = {len(PARTIAL_SWEEP_PARAMETERS) * REPEATS_PER_COMBO} total runs")
        
        for _ in range(REPEATS_PER_COMBO):
            for param in PARTIAL_SWEEP_PARAMETERS:
                run_id_counter += 1
                overrides = { "K": param[0], "P": param[1] }
                all_overrides.append((overrides, run_id_counter))
    
    elif args.mode == 'multi-game-sweep':
        # Multi-game sweep: optimal K/P combinations for different games
        # Each dictionary contains ENV_NAME, K, and P values
        print(f"Multi-game sweep: {len(MULTI_GAME_SWEEP_GAME_CONFIGS)} game configurations x {REPEATS_PER_COMBO} repeats = {len(MULTI_GAME_SWEEP_GAME_CONFIGS) * REPEATS_PER_COMBO} total runs")
        for i, config in enumerate(MULTI_GAME_SWEEP_GAME_CONFIGS, 1):
            print(f"  {i}. {config['ENV_NAME']} (K={config['K']}, P={config['P']})")
        
        for _ in range(REPEATS_PER_COMBO):
            for config in MULTI_GAME_SWEEP_GAME_CONFIGS:
                run_id_counter += 1
                overrides = {
                    "ENV_NAME": config["ENV_NAME"],
                    "K": config["K"], 
                    "P": config["P"]
                }
                all_overrides.append((overrides, run_id_counter))

    # Launch Ray tasks
    ray_tasks = []
    for overrides, rid in all_overrides:
        task = run_benchmark_with_overrides.remote(overrides, rid)
        ray_tasks.append(task)

    # As tasks finish, push each into SQLite immediately
    remaining = set(ray_tasks)
    while remaining:
        # Wait for at least one to finish
        done, remaining = ray.wait(list(remaining), num_returns=1, timeout=None)
        finished_ref = done[0]
        run_data = ray.get(finished_ref)

        # run_data is a dict with keys: "run_id", "overrides", "games"
        rid = run_data["run_id"]
        overrides = run_data["overrides"]       # e.g. {"K":100, "P":0.9}
        games_dict = run_data["games"]          # {"Pong-v5": {...}, "Breakout-v5": {...}, …}

        # JSON‐serialize overrides once
        overrides_json = json.dumps(overrides)

        # For each game in this run, INSERT one row
        for game_name, game_data in games_dict.items():
            fitness_log = game_data["fitness_log"]              # list of [g, i, ep, rew]
            best_indivs = game_data["best_individuals"]         # list of dicts
            plot_data   = game_data["plot_data"]                # list of [g, best, avg]

            # Convert NumPy→list already done in the remote call (solutions). So just dump:
            fitness_log_json      = json.dumps(fitness_log)
            best_indivs_json      = json.dumps(best_indivs)
            plot_data_json        = json.dumps(plot_data)

            total_generations     = len(plot_data)
            # best_fitness is the last generation's "best":
            best_fitness          = float(plot_data[-1][1]) if total_generations > 0 else 0.0

            insert_sql = """
                INSERT INTO runs
                    (run_id, game, overrides_json,
                     fitness_log_json, best_individuals_json, plot_data_json,
                     total_generations, best_fitness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.execute(
                insert_sql,
                (
                    rid,
                    game_name,
                    overrides_json,
                    fitness_log_json,
                    best_indivs_json,
                    plot_data_json,
                    total_generations,
                    best_fitness,
                )
            )
            conn.commit()

        print(f"⇢ Pushed run_id={rid} into DB (games: {list(games_dict.keys())})")

    # All tasks are done. Close DB & Ray.
    cursor.close()
    conn.close()
    ray.shutdown()

    print(f"\nAll runs completed and written to '{DB_PATH}'.")


if __name__ == "__main__":
    main()
