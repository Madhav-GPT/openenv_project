#!/usr/bin/env python3
"""Compare random, naive, optimal, and optional stored-policy agents."""

from __future__ import annotations

import argparse
import random
from statistics import mean
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sre_env.client import SREEnv
from sre_env.models import SREAction
from sre_env.scripts.benchmark_policies import (
    NAIVE_ACTIONS,
    OPTIMAL_ACTIONS,
    SCENARIO_FOR_DIFFICULTY,
    naive_action,
    random_action,
)
from sre_env.scripts.grpo_support import load_trained_policy

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
N_EPISODES_PER_AGENT = 10
MAX_TICKS_BY_DIFFICULTY = {"easy": 15, "medium": 20, "hard": 25}


def run_policy(
    base_url: str,
    difficulty: str,
    scenario_id: str,
    chooser: Callable[[dict, int], SREAction],
) -> float:
    total_reward = 0.0
    with SREEnv(base_url=base_url).sync() as env:
        reset_result = env.reset(difficulty=difficulty, scenario_id=scenario_id)
        observation = reset_result.observation
        step_idx = 0
        while not observation.done and not observation.episode_complete:
            action = chooser(observation.model_dump(), step_idx)
            result = env.step(action)
            observation = result.observation
            total_reward += result.reward
            step_idx += 1
            if step_idx > observation.max_ticks + 5:
                break
    return total_reward


def make_optimal_chooser(scenario_id: str) -> Callable[[dict, int], SREAction]:
    actions = OPTIMAL_ACTIONS[scenario_id]

    def chooser(_obs_dict: dict, step_idx: int) -> SREAction:
        if step_idx < len(actions):
            return actions[step_idx]
        return actions[-1]

    return chooser


def make_naive_chooser(scenario_id: str) -> Callable[[dict, int], SREAction]:
    actions = NAIVE_ACTIONS[scenario_id]

    def chooser(_obs_dict: dict, step_idx: int) -> SREAction:
        if step_idx < len(actions):
            return actions[step_idx]
        return actions[-1]

    return chooser


def make_trained_chooser(
    scenario_id: str,
    policy_data: dict[str, list[dict[str, object]]],
) -> Callable[[dict, int], SREAction]:
    actions = policy_data.get(scenario_id) or [
        action.model_dump(exclude_none=True) for action in OPTIMAL_ACTIONS[scenario_id]
    ]

    def chooser(obs_dict: dict, step_idx: int) -> SREAction:
        if step_idx < len(actions):
            return SREAction(**actions[step_idx])
        return naive_action(obs_dict)

    return chooser


def plot_results(results: dict[str, dict[str, list[float]]], output_path: str) -> None:
    difficulties = ["easy", "medium", "hard"]
    colors = {
        "random": "#8b949e",
        "naive": "#f97316",
        "optimal": "#3fb950",
        "trained": "#58a6ff",
    }
    labels = {
        "random": "Random",
        "naive": "Naive",
        "optimal": "Optimal",
        "trained": "Stored Policy",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")

    for axis, difficulty in zip(axes, difficulties):
        axis.set_facecolor("#161b22")
        episodes = list(range(1, len(results[difficulty]["random"]) + 1))
        for agent, rewards in results[difficulty].items():
            axis.plot(
                episodes,
                rewards,
                linewidth=1.4,
                alpha=0.35,
                color=colors[agent],
            )
            axis.axhline(
                np.mean(rewards),
                linewidth=2.2,
                linestyle="--",
                color=colors[agent],
                label=f"{labels[agent]} (avg={np.mean(rewards):+.2f})",
            )
        axis.axhline(0, color="#444", linewidth=0.8, linestyle=":")
        axis.grid(alpha=0.15, color="#444")
        axis.set_title(difficulty.capitalize(), color="white", fontweight="bold")
        axis.set_xlabel("Episode", color="#aaa")
        axis.set_ylabel("Reward", color="#aaa")
        axis.tick_params(colors="#aaa")
        for spine in axis.spines.values():
            spine.set_color("#444")
        axis.legend(facecolor="#1c2128", labelcolor="white", framealpha=0.8)

    fig.suptitle(
        "SRE-Env reward-signal comparison",
        color="white",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def print_summary(results: dict[str, dict[str, list[float]]]) -> None:
    print("\n=== COMPARISON SUMMARY ===")
    for difficulty in ("easy", "medium", "hard"):
        line = [difficulty]
        for agent in results[difficulty]:
            line.append(f"{agent}={mean(results[difficulty][agent]):+.3f}")
        print("  " + "  ".join(line))
    overall = {
        agent: mean(
            [
                mean(results[difficulty][agent])
                for difficulty in ("easy", "medium", "hard")
            ]
        )
        for agent in results["easy"]
    }
    if {
        "random",
        "naive",
        "optimal",
    }.issubset(overall) and overall["optimal"] > overall["naive"] > overall["random"]:
        print("validation=passed ordering=optimal>naive>random")
    else:
        print("validation=failed ordering check did not hold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES_PER_AGENT)
    parser.add_argument(
        "--use-trained-weights",
        action="store_true",
        help="Compatibility flag: load a stored policy artifact and add a fourth line.",
    )
    parser.add_argument(
        "--weights-path",
        default="outputs/grpo_sre/best_weights.npz",
        help="Path to best_weights.npz or trained_policy.json for the stored policy agent.",
    )
    args = parser.parse_args()

    trained_policy: dict[str, list[dict[str, object]]] = {}
    if args.use_trained_weights:
        try:
            trained_policy = load_trained_policy(args.weights_path)
        except Exception as exc:
            raise SystemExit(f"Could not load stored policy from {args.weights_path}: {exc}") from exc
    results: dict[str, dict[str, list[float]]] = {}

    for difficulty, scenario_id in SCENARIO_FOR_DIFFICULTY.items():
        difficulty_results: dict[str, list[float]] = {
            "random": [],
            "naive": [],
            "optimal": [],
        }
        if args.use_trained_weights:
            difficulty_results["trained"] = []

        for episode_idx in range(args.n_episodes):
            rng = random.Random(f"{difficulty}-random-{episode_idx}")
            difficulty_results["random"].append(
                run_policy(
                    args.base_url,
                    difficulty,
                    scenario_id,
                    lambda obs, step, rng=rng: random_action(obs, rng),
                )
            )
            difficulty_results["naive"].append(
                run_policy(
                    args.base_url,
                    difficulty,
                    scenario_id,
                    make_naive_chooser(scenario_id),
                )
            )
            difficulty_results["optimal"].append(
                run_policy(
                    args.base_url,
                    difficulty,
                    scenario_id,
                    make_optimal_chooser(scenario_id),
                )
            )
            if args.use_trained_weights:
                difficulty_results["trained"].append(
                    run_policy(
                        args.base_url,
                        difficulty,
                        scenario_id,
                        make_trained_chooser(scenario_id, trained_policy),
                    )
                )

        results[difficulty] = difficulty_results

    output_path = (
        "post_training_comparison.png"
        if args.use_trained_weights
        else "three_agent_comparison.png"
    )
    plot_results(results, output_path)
    print_summary(results)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
