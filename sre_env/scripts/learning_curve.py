#!/usr/bin/env python3
"""Plot baseline vs few-shot performance for the SRE environment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sre_env.client import SREEnv
from sre_env.models import SREAction
from sre_env.scripts.live_dashboard import LearningCurveDashboard
from sre_env.scripts.llm_backends import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    LLMCallMetrics,
    LLMCallResult,
    LLMConfig,
    build_llm_config,
    call_action_model,
)

N_EPISODES = 20
OUTPUT_PATH = Path("learning_curve.png")
SYSTEM_BASELINE = """You are an SRE responding to a live production incident.
Use only JSON tool calls.
Investigate before fixing and fix the root cause first."""


def _system_fewshot(history_text: str) -> str:
    return f"""You are an SRE improving from past experience.
Use only JSON tool calls.
Investigate before fixing and fix the root cause first.

Past episode outcomes:
{history_text}
"""


def _key(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("tool"),
        action.get("service"),
        action.get("metric"),
        action.get("replicas"),
        action.get("version"),
    )


def _baseline_plan(difficulty: str) -> list[dict[str, Any]]:
    if difficulty == "easy":
        return [
            {"tool": "restart", "service": "api-gateway"},
            {"tool": "get_logs", "service": "database"},
            {"tool": "restart", "service": "database"},
        ]
    if difficulty == "medium":
        return [
            {"tool": "restart", "service": "database"},
            {"tool": "get_logs", "service": "cache"},
            {"tool": "restart", "service": "cache"},
            {"tool": "restart", "service": "database"},
        ]
    return [
        {"tool": "restart", "service": "database"},
        {"tool": "get_metrics", "service": "worker", "metric": "memory"},
        {"tool": "get_logs", "service": "worker"},
        {"tool": "rollback", "service": "worker", "version": "previous"},
        {"tool": "restart", "service": "database"},
        {"tool": "restart", "service": "api-gateway"},
    ]


def _fewshot_exploration_plan(difficulty: str) -> list[dict[str, Any]]:
    if difficulty == "easy":
        return [
            {"tool": "get_logs", "service": "database"},
            {"tool": "restart", "service": "database"},
        ]
    if difficulty == "medium":
        return [
            {"tool": "get_logs", "service": "cache"},
            {"tool": "get_logs", "service": "api-gateway"},
            {"tool": "restart", "service": "cache"},
            {"tool": "restart", "service": "database"},
        ]
    return [
        {"tool": "get_metrics", "service": "worker", "metric": "memory"},
        {"tool": "get_logs", "service": "worker"},
        {"tool": "get_logs", "service": "database"},
        {"tool": "rollback", "service": "worker", "version": "previous"},
        {"tool": "restart", "service": "database"},
        {"tool": "restart", "service": "api-gateway"},
    ]


def _obs_to_text(observation: dict[str, Any]) -> str:
    lines = [f"TICK {observation['tick']}/{observation['max_ticks']}"]
    alerts = observation.get("active_alerts", [])
    if alerts:
        lines.append("ACTIVE ALERTS:")
        for alert in alerts:
            lines.append(
                f"- [{alert['severity'].upper()}] {alert['service']}: {alert['message']}"
            )
    else:
        lines.append("ACTIVE ALERTS: none")
    lines.append("SERVICES:")
    for name, service in observation["services"].items():
        lines.append(
            f"- {name}: {service['status']} cpu={service['cpu_pct']} mem={service['memory_pct']} err={service['error_rate_pct']}"
        )
    if observation.get("last_action_result"):
        lines.append(f"LAST RESULT: {observation['last_action_result']}")
    if observation.get("tool_output"):
        lines.append(f"TOOL OUTPUT: {observation['tool_output']}")
    lines.append("Next action. JSON only.")
    return "\n".join(lines)


def _choose_action(
    difficulty: str,
    history: list[dict[str, Any]],
    memory: list[dict[str, Any]] | None,
    rng: random.Random,
    mode: str,
) -> dict[str, Any]:
    seen = {_key(action) for action in history}

    if mode == "fewshot" and memory:
        for action in memory:
            if _key(action) not in seen:
                return action

    plan = _fewshot_exploration_plan(difficulty) if mode == "fewshot" else _baseline_plan(difficulty)
    if mode == "baseline" and history and rng.random() < 0.15:
        return {"tool": "get_dependencies", "service": "api-gateway"}
    for action in plan:
        if _key(action) not in seen:
            return action
    return plan[-1]


def _run_episode(
    difficulty: str,
    base_url: str,
    episode_idx: int,
    memory: list[dict[str, Any]] | None,
    mode: str,
    llm_config: LLMConfig,
    dashboard: LearningCurveDashboard,
    prior_summaries: list[str] | None = None,
) -> tuple[float, bool, list[dict[str, Any]]]:
    total_reward = 0.0
    actions: list[dict[str, Any]] = []
    rng = random.Random(f"{difficulty}-{mode}-{episode_idx}")
    if llm_config.provider == "heuristic":
        messages: list[dict[str, str]] | None = None
    else:
        if mode == "fewshot":
            history_text = "\n".join((prior_summaries or [])[-3:]) or "No history yet."
            system_prompt = _system_fewshot(history_text)
        else:
            system_prompt = SYSTEM_BASELINE
        messages = [{"role": "system", "content": system_prompt}]

    dashboard.start_run(difficulty=difficulty, mode=mode, episode_index=episode_idx + 1)
    with SREEnv(base_url=base_url).sync() as env:
        reset_result = env.reset(difficulty=difficulty)
        observation = reset_result.observation
        done = reset_result.done
        while not done:
            pre_observation = observation.model_dump()
            fallback = _choose_action(difficulty, actions, memory, rng, mode)
            if messages is None:
                call_result = LLMCallResult(
                    action=fallback,
                    metrics=LLMCallMetrics(
                        provider="heuristic",
                        model=llm_config.model,
                        raw_preview="deterministic heuristic policy",
                    ),
                )
            else:
                obs_text = _obs_to_text(observation.model_dump())
                messages.append({"role": "user", "content": obs_text})
                call_result = call_action_model(
                    llm_config,
                    messages,
                    fallback,
                    temperature=0.2,
                    max_tokens=120,
                )
                messages.append({"role": "assistant", "content": json.dumps(call_result.action)})
            try:
                action = SREAction(**call_result.action)
            except Exception as exc:
                call_result.metrics.mark_fallback(f"schema validation: {exc}")
                action = SREAction(**fallback)
            action_dict = action.model_dump(exclude_none=True)
            actions.append(action_dict)
            result = env.step(action)
            observation = result.observation
            total_reward += result.reward
            done = result.done
            dashboard.record_step(
                difficulty=difficulty,
                mode=mode,
                action=action_dict,
                pre_observation=pre_observation,
                observation=observation.model_dump(),
                reward=result.reward,
                total_reward=total_reward,
                metrics=call_result.metrics,
            )
        dashboard.finish_episode(
            difficulty=difficulty,
            mode=mode,
            reward=total_reward,
            success=observation.success,
            failure_reason=observation.failure_reason,
        )
        return total_reward, observation.success, actions


def _smooth(values: list[float], window: int = 3) -> list[float]:
    output = []
    for idx in range(len(values)):
        lo = max(0, idx - window // 2)
        hi = min(len(values), idx + window // 2 + 1)
        output.append(float(np.mean(values[lo:hi])))
    return output


def _episode_summary(
    difficulty: str,
    reward: float,
    success: bool,
    actions: list[dict[str, Any]],
) -> str:
    recent = [f"{item.get('tool')}({item.get('service')})" for item in actions[-3:]]
    return (
        f"difficulty={difficulty} reward={reward:.2f} success={success} "
        f"actions={len(actions)} recent={recent}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=SREEnv.DEFAULT_BASE_URL)
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "ollama", "groq", "heuristic"],
    )
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Also save the post-run matplotlib learning-curve image.",
    )
    parser.add_argument(
        "--plot-path",
        default=str(OUTPUT_PATH),
        help="Output path for the optional saved plot.",
    )
    args = parser.parse_args()

    llm_config = build_llm_config(
        provider=args.provider,
        model=args.model,
        ollama_host=args.ollama_host,
    )

    difficulties = ("easy", "medium", "hard")
    results: dict[str, tuple[list[float], list[float]]] = {}

    with LearningCurveDashboard(
        provider=llm_config.provider,
        model=llm_config.model,
        total_episodes=args.episodes,
    ) as dashboard:
        for difficulty in difficulties:
            baseline_rewards: list[float] = []
            fewshot_rewards: list[float] = []
            learned_plan: list[dict[str, Any]] | None = None
            fewshot_history: list[str] = []

            for episode_idx in range(args.episodes):
                base_reward, _, _ = _run_episode(
                    difficulty=difficulty,
                    base_url=args.base_url,
                    episode_idx=episode_idx,
                    memory=None,
                    mode="baseline",
                    llm_config=llm_config,
                    dashboard=dashboard,
                )
                baseline_rewards.append(base_reward)

                few_reward, success, actions = _run_episode(
                    difficulty=difficulty,
                    base_url=args.base_url,
                    episode_idx=episode_idx,
                    memory=learned_plan,
                    mode="fewshot",
                    llm_config=llm_config,
                    dashboard=dashboard,
                    prior_summaries=fewshot_history,
                )
                fewshot_rewards.append(few_reward)
                if success:
                    learned_plan = actions
                fewshot_history.append(
                    _episode_summary(difficulty, few_reward, success, actions)
                )

            results[difficulty] = (baseline_rewards, fewshot_rewards)

    plot_path = Path(args.plot_path)
    if args.save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#0d1117")
        colors = {"baseline": "#ff6b6b", "fewshot": "#4ecdc4"}
        titles = {
            "easy": "Easy - DB Crash",
            "medium": "Medium - Cache Cascade",
            "hard": "Hard - Bad Deploy",
        }

        for axis, difficulty in zip(axes, difficulties):
            baseline_rewards, fewshot_rewards = results[difficulty]
            episodes = list(range(1, len(baseline_rewards) + 1))

            axis.set_facecolor("#161b22")
            axis.plot(
                episodes,
                baseline_rewards,
                color=colors["baseline"],
                alpha=0.25,
                linewidth=1,
            )
            axis.plot(
                episodes,
                fewshot_rewards,
                color=colors["fewshot"],
                alpha=0.25,
                linewidth=1,
            )
            axis.plot(
                episodes,
                _smooth(baseline_rewards),
                color=colors["baseline"],
                linewidth=2.5,
                label=f"Baseline (avg={np.mean(baseline_rewards):.2f})",
            )
            axis.plot(
                episodes,
                _smooth(fewshot_rewards),
                color=colors["fewshot"],
                linewidth=2.5,
                label=f"Few-shot (avg={np.mean(fewshot_rewards):.2f})",
            )
            axis.axhline(0, color="#444", linestyle="--", linewidth=0.8)
            axis.set_title(titles[difficulty], color="white", fontsize=13, fontweight="bold")
            axis.set_xlabel("Episode", color="#aaa")
            axis.set_ylabel("Total reward", color="#aaa")
            axis.tick_params(colors="#aaa")
            for spine in axis.spines.values():
                spine.set_color("#444")
            axis.grid(alpha=0.15, color="#444")
            axis.legend(facecolor="#1c2128", labelcolor="white", framealpha=0.8)

        fig.suptitle(
            "SRE-Env: Baseline vs Few-shot Agent Learning Curve",
            color="white",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(
        f"provider={llm_config.provider} model={llm_config.model} "
        f"base_url={args.base_url} episodes={args.episodes}"
    )
    for difficulty in difficulties:
        baseline_rewards, fewshot_rewards = results[difficulty]
        print(
            f"{difficulty}: baseline_avg={np.mean(baseline_rewards):+.2f} "
            f"fewshot_avg={np.mean(fewshot_rewards):+.2f}"
        )
    if args.save_plot:
        print(f"saved={plot_path}")


if __name__ == "__main__":
    main()
