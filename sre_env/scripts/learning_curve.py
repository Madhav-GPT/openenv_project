#!/usr/bin/env python3
"""Phase 2 learning-curve runner backed by the unified dashboard flow."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.live import Live

from sre_env.scripts.learning_curve_dashboard import (
    DEFAULT_BASE_URL,
    SCENARIO_FOR_DIFFICULTY,
    DashboardState,
    make_baseline_policy,
    make_reference_policy,
    render_dashboard,
    run_agent_episode,
)

OUTPUT_PATH = Path("learning_curve.png")


def smooth(values: list[float], window: int = 3) -> list[float]:
    if not values:
        return []
    output: list[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - window // 2)
        hi = min(len(values), idx + window // 2 + 1)
        output.append(float(np.mean(values[lo:hi])))
    return output


def save_plot(results: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    difficulties = ["easy", "medium", "hard"]
    colors = {"baseline": "#ff6b6b", "reference": "#4ecdc4"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")

    for axis, difficulty in zip(axes, difficulties):
        baseline_rewards = results[difficulty]["baseline"]
        reference_rewards = results[difficulty]["reference"]
        episodes = list(range(1, len(baseline_rewards) + 1))

        axis.set_facecolor("#161b22")
        axis.plot(
            episodes,
            baseline_rewards,
            color=colors["baseline"],
            linewidth=1.2,
            alpha=0.35,
        )
        axis.plot(
            episodes,
            reference_rewards,
            color=colors["reference"],
            linewidth=1.2,
            alpha=0.35,
        )
        axis.plot(
            episodes,
            smooth(baseline_rewards),
            color=colors["baseline"],
            linewidth=2.2,
            label=f"Baseline (avg={np.mean(baseline_rewards):+.2f})",
        )
        axis.plot(
            episodes,
            smooth(reference_rewards),
            color=colors["reference"],
            linewidth=2.2,
            label=f"Reference (avg={np.mean(reference_rewards):+.2f})",
        )
        axis.axhline(0, color="#444", linestyle=":", linewidth=0.8)
        axis.grid(alpha=0.15, color="#444")
        axis.set_title(difficulty.capitalize(), color="white", fontweight="bold")
        axis.set_xlabel("Episode", color="#aaa")
        axis.set_ylabel("Reward", color="#aaa")
        axis.tick_params(colors="#aaa")
        for spine in axis.spines.values():
            spine.set_color("#444")
        axis.legend(facecolor="#1c2128", labelcolor="white", framealpha=0.8)

    fig.suptitle(
        "SRE-Env Phase 2 learning curve",
        color="white",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Baseline vs reference trajectories on the unified three-phase environment.",
        ha="center",
        color="#8b949e",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_learning_curve(
    *,
    base_url: str,
    provider: str,
    model: str,
    episodes: int,
) -> tuple[DashboardState, dict[str, dict[str, list[float]]]]:
    state = DashboardState(
        provider=provider,
        student_model=model,
        episodes=episodes,
        difficulties=["easy", "medium", "hard"],
    )
    results = {
        difficulty: {"baseline": [], "reference": []}
        for difficulty in state.difficulties
    }
    baseline_policy = make_baseline_policy()
    reference_policy = make_reference_policy()

    with Live(render_dashboard(state), refresh_per_second=4) as live:
        for difficulty in state.difficulties:
            scenario_id = SCENARIO_FOR_DIFFICULTY[difficulty]
            for episode_idx in range(1, episodes + 1):
                state.current_difficulty = difficulty
                state.current_episode = episode_idx

                baseline_reward, _ = run_agent_episode(
                    base_url=base_url,
                    difficulty=difficulty,
                    scenario_id=scenario_id,
                    agent_name="baseline",
                    chooser=baseline_policy,
                    state=state,
                    live=live,
                )
                state.baseline_reward_history.append(baseline_reward)
                results[difficulty]["baseline"].append(baseline_reward)

                reference_reward, _ = run_agent_episode(
                    base_url=base_url,
                    difficulty=difficulty,
                    scenario_id=scenario_id,
                    agent_name="reference",
                    chooser=reference_policy,
                    state=state,
                    live=live,
                )
                state.reference_reward_history.append(reference_reward)
                results[difficulty]["reference"].append(reference_reward)
                state.log_lines.append(
                    f"completed difficulty={difficulty} episode={episode_idx} baseline={baseline_reward:+.2f} reference={reference_reward:+.2f}"
                )
                live.update(render_dashboard(state), refresh=True)

    return state, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--save-plot", action="store_true")
    args = parser.parse_args()

    _, results = run_learning_curve(
        base_url=args.base_url,
        provider=args.provider,
        model=args.model,
        episodes=args.episodes,
    )

    for difficulty in ("easy", "medium", "hard"):
        baseline_avg = np.mean(results[difficulty]["baseline"])
        reference_avg = np.mean(results[difficulty]["reference"])
        print(
            f"difficulty={difficulty} baseline_avg={baseline_avg:+.3f} "
            f"reference_avg={reference_avg:+.3f}"
        )

    if args.save_plot:
        save_plot(results, OUTPUT_PATH)
        print(f"saved={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
