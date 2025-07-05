"""
Single run of the SCOPE policy for testing purposes
"""

import numpy as np
import cma
import gymnasium as gym
import ale_py

from SCOPE import *


# Simple config information for debugging
CONFIG = {
    "ENV_NAME": "ALE/SpaceInvaders-v5",
    "EPISODES_PER_INDIVIDUAL": 1,
    "MAX_STEPS_PER_EPISODE": 10000,
    "POPULATION_SIZE": None,
    "CMA_SIGMA": 0.5,
    "GENERATIONS": 5000,
    "K": 75,
    "P": 25,
    "REPEAT_ACTION_PROBABILITY": 0.0,
    "FRAMESKIP": 4
}


def make_env(game_name: str,
             repeat_action_prob: float,
             frameskip: int) -> gym.Env:
    """Create a Gym environment for the given game"""
    return gym.make(id=game_name,
                    obs_type="grayscale",
                    repeat_action_probability=repeat_action_prob,
                    frameskip=frameskip)


def evaluate_individual(solution: list,
                        game: str,
                        output_size: int,
                        config: dict) -> float:
    """Evaluate an individual policy"""
    # Create the policy
    policy = SCOPE(chromosome=solution,
                   k=config["K"],
                   p=config["P"],
                   output_size=output_size)

    # Create the environment
    env = make_env(game_name=game,
                   repeat_action_prob=config["REPEAT_ACTION_PROBABILITY"],
                   frameskip=config["FRAMESKIP"])
    
    total_reward = 0.0

    for _ in range(config["EPISODES_PER_INDIVIDUAL"]):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < config["MAX_STEPS_PER_EPISODE"]:
            state = obs.astype(np.float32) / 255.0
            scope_output = policy.forward(state)
            action = int(np.argmax(scope_output))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        total_reward += ep_reward

    env.close()
    return total_reward / config["EPISODES_PER_INDIVIDUAL"]  # Return the average reward


def main():
    """Run the SCOPE policy"""
    game = CONFIG["ENV_NAME"]
    env = make_env(game_name=game,
                   repeat_action_prob=CONFIG["REPEAT_ACTION_PROBABILITY"],
                   frameskip=CONFIG["FRAMESKIP"])
    output_size = env.action_space.n
    chromosome_size = compute_chromosome_size(k=CONFIG["K"], output_size=output_size)
    env.close()

    es = cma.CMAEvolutionStrategy(x0=np.zeros(chromosome_size),
                                  sigma0=CONFIG["CMA_SIGMA"],
                                  inopts={"popsize": CONFIG["POPULATION_SIZE"]})

    best_overall_reward = float("-inf")

    for generation in range(CONFIG["GENERATIONS"]):
        solutions = es.ask()
        rewards = []

        for index, solution in enumerate(solutions):
            avg_reward = evaluate_individual(solution, game, output_size, CONFIG)
            rewards.append(-avg_reward)  # Negative fitness as CMA-ES minimizes

            if avg_reward > best_overall_reward:
                best_overall_reward = avg_reward

            print(f"[GEN {generation+1}] Indv {index+1}/{len(solutions)}: Reward = {avg_reward:.2f}")

        es.tell(solutions, rewards)

        best_gen_reward = -min(rewards)
        avg_gen_reward = -np.mean(rewards)
        print(f"[GEN {generation+1}] Best: {best_gen_reward:.2f} | Avg: {avg_gen_reward:.2f} | Best Overall: {best_overall_reward:.2f}")


if __name__ == "__main__":
    main()
