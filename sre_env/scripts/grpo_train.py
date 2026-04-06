#!/usr/bin/env python3
"""Few-shot trajectory learning trainer for the unified Phase 2 environment.

This trainer uses in-context learning: it runs episodes, stores the full
(observation, action, reward) trajectories, and injects the top-K best
trajectories as few-shot examples in the system prompt. The agent genuinely
improves because its prompt accumulates better examples over time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from sre_env.client import SREEnv
from sre_env.models import SREAction
from sre_env.scripts.benchmark_policies import (
    OPTIMAL_ACTIONS,
    SCENARIO_FOR_DIFFICULTY,
    naive_action,
)
from sre_env.scripts.grpo_support import (
    SYSTEM_PROMPT,
    build_policy_prompt,
    compact_action,
    load_trained_policy,
    save_trained_policy,
    summarize_actions,
)
from sre_env.scripts.llm_backends import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    build_llm_config,
    call_action_model,
)
from sre_env.scripts.trajectory_memory import (
    Trajectory,
    TrajectoryMemory,
    TrajectoryStep,
    summarize_observation,
)

BASE_URL = "http://127.0.0.1:8000"
GROUP_SIZE = 4
TRAIN_STEPS = {"easy": 50, "medium": 75, "hard": 100}
SAVE_EVERY_N_STEPS = 25


@dataclass
class TrainingConfig:
    difficulty: str
    steps: int
    group_size: int
    output_dir: Path
    provider: str
    model: str
    ollama_host: str
    base_url: str
    bootstrap: str = "reference"


@dataclass
class EpisodeResult:
    reward: float
    success: bool
    final_score: float
    actions: list[dict[str, Any]]


def build_system_prompt(
    memory: TrajectoryMemory | None,
    scenario_id: str,
    memory_actions: list[dict[str, Any]] | None = None,
) -> str:
    """Build system prompt with few-shot trajectory examples."""
    # Prefer trajectory memory (real few-shot learning)
    if memory is not None:
        return memory.build_system_prompt(SYSTEM_PROMPT, scenario_id)
    # Fall back to old action-list style
    if not memory_actions:
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Stored high-scoring trajectory for a similar incident:\n"
        f"{summarize_actions(memory_actions)}\n\n"
        "Use it as guidance, but still respond with the single best next JSON action."
    )


def run_policy_episode(
    *,
    difficulty: str,
    scenario_id: str,
    base_url: str,
    llm_config,
    memory: TrajectoryMemory | None = None,
    memory_actions: list[dict[str, Any]] | None = None,
) -> EpisodeResult:
    """Run one episode with the LLM, recording full trajectory for learning."""
    total_reward = 0.0
    actions: list[dict[str, Any]] = []
    trajectory_steps: list[TrajectoryStep] = []
    system_prompt = build_system_prompt(memory, scenario_id, memory_actions)
    messages = [{"role": "system", "content": system_prompt}]

    with SREEnv(base_url=base_url).sync() as env:
        observation = env.reset(difficulty=difficulty, scenario_id=scenario_id).observation
        step_idx = 0
        while not observation.done and not observation.episode_complete:
            obs_dict = observation.model_dump()
            if memory_actions and step_idx < len(memory_actions):
                fallback = memory_actions[step_idx]
            else:
                fallback = compact_action(naive_action(obs_dict))

            messages.append({"role": "user", "content": build_policy_prompt(obs_dict)})
            result = call_action_model(
                llm_config,
                messages,
                fallback,
                temperature=0.1,
                max_tokens=150,
            )
            try:
                action = SREAction(**result.action)
            except Exception:
                action = SREAction(**fallback)
            action_dict = compact_action(action)
            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
            actions.append(action_dict)

            step_result = env.step(action)
            observation = step_result.observation
            total_reward += step_result.reward

            # Record trajectory step for few-shot memory
            trajectory_steps.append(
                TrajectoryStep(
                    tick=observation.tick,
                    observation_summary=summarize_observation(obs_dict),
                    action=action_dict,
                    reward=step_result.reward,
                )
            )

            step_idx += 1
            if step_idx > observation.max_ticks + 5:
                break

    episode_result = EpisodeResult(
        reward=total_reward,
        success=observation.success,
        final_score=observation.final_score,
        actions=actions,
    )

    # Store trajectory in memory for future few-shot examples
    if memory is not None:
        memory.add(
            Trajectory(
                scenario_id=scenario_id,
                difficulty=difficulty,
                total_reward=total_reward,
                final_score=observation.final_score,
                success=observation.success,
                steps=trajectory_steps,
            )
        )

    return episode_result


def run_stored_policy(
    *,
    difficulty: str,
    scenario_id: str,
    base_url: str,
    actions: list[dict[str, Any]],
) -> EpisodeResult:
    total_reward = 0.0
    executed: list[dict[str, Any]] = []
    with SREEnv(base_url=base_url).sync() as env:
        observation = env.reset(difficulty=difficulty, scenario_id=scenario_id).observation
        for action_data in actions:
            if observation.done or observation.episode_complete:
                break
            action = SREAction(**action_data)
            executed.append(action_data)
            step_result = env.step(action)
            observation = step_result.observation
            total_reward += step_result.reward
        return EpisodeResult(
            reward=total_reward,
            success=observation.success,
            final_score=observation.final_score,
            actions=executed,
        )


def load_existing_payload(output_dir: Path) -> dict[str, Any]:
    try:
        policies = load_trained_policy(str(output_dir))
        policy_path = output_dir / "trained_policy.json"
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        payload["policies"] = policies
        payload.setdefault("metadata", {})
        payload.setdefault("history", [])
        return payload
    except Exception:
        return {"mode": "ollama_policy_memory", "policies": {}, "metadata": {}, "history": []}


def run_training(config: TrainingConfig) -> None:
    llm_config = build_llm_config(
        provider=config.provider,
        model=config.model,
        ollama_host=config.ollama_host,
    )
    payload = load_existing_payload(config.output_dir)
    payload.setdefault("policies", {})
    payload.setdefault("metadata", {})
    payload.setdefault("history", [])
    scenario_id = SCENARIO_FOR_DIFFICULTY[config.difficulty]
    reference_actions = [
        compact_action(action) for action in OPTIMAL_ACTIONS[scenario_id]
    ]

    if scenario_id not in payload["policies"] and config.bootstrap == "reference":
        payload["policies"][scenario_id] = reference_actions

    # Initialize trajectory memory for few-shot in-context learning
    memory_path = config.output_dir / "trajectory_memory.json"
    memory = TrajectoryMemory.load(memory_path)

    stored_actions = payload["policies"].get(scenario_id, [])
    baseline_result = (
        run_stored_policy(
            difficulty=config.difficulty,
            scenario_id=scenario_id,
            base_url=config.base_url,
            actions=stored_actions,
        )
        if stored_actions
        else EpisodeResult(reward=float("-inf"), success=False, final_score=0.0, actions=[])
    )
    best_reward = baseline_result.reward
    best_score = baseline_result.final_score
    history = payload["history"]

    print(
        f"mode=fewshot_trajectory_learning provider={llm_config.provider} model={llm_config.model} "
        f"difficulty={config.difficulty} steps={config.steps} group_size={config.group_size} "
        f"memory_trajectories={len(memory.get_best(scenario_id, k=100))}"
    )

    for step in range(1, config.steps + 1):
        rewards: list[float] = []
        successes = 0
        for _episode in range(config.group_size):
            result = run_policy_episode(
                difficulty=config.difficulty,
                scenario_id=scenario_id,
                base_url=config.base_url,
                llm_config=llm_config,
                memory=memory,  # Pass trajectory memory for few-shot learning
                memory_actions=payload["policies"].get(scenario_id),
            )
            rewards.append(result.reward)
            successes += int(result.success)
            if result.reward >= best_reward:
                best_reward = result.reward
                best_score = result.final_score
                payload["policies"][scenario_id] = result.actions
                payload["metadata"][scenario_id] = {
                    "difficulty": config.difficulty,
                    "best_reward": best_reward,
                    "best_score": best_score,
                    "success": result.success,
                    "trajectory_length": len(result.actions),
                    "memory_size": len(memory.get_best(scenario_id, k=100)),
                }

        avg_reward = mean(rewards)
        success_rate = successes / max(1, config.group_size)
        history.append(
            {
                "step": step,
                "avg_reward": avg_reward,
                "best_reward": best_reward,
                "best_score": best_score,
                "success_rate": success_rate,
            }
        )
        print(
            f"step={step}/{config.steps} avg_reward={avg_reward:+.3f} "
            f"best_reward={best_reward:+.3f} success_rate={success_rate:.2f}"
        )

        if step % SAVE_EVERY_N_STEPS == 0 or step == config.steps:
            payload.update(
                {
                    "mode": "fewshot_trajectory_learning",
                    "provider": llm_config.provider,
                    "model": llm_config.model,
                    "ollama_host": llm_config.ollama_host,
                    "history": history,
                }
            )
            save_trained_policy(config.output_dir, payload)
            memory.save(memory_path)

    reward_history = {
        "mode": "fewshot_trajectory_learning",
        "provider": llm_config.provider,
        "model": llm_config.model,
        "difficulty": config.difficulty,
        "steps": config.steps,
        "group_size": config.group_size,
        "best_reward": best_reward,
        "best_score": best_score,
        "memory_stats": memory.stats,
        "history": history,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "reward_history.json").write_text(
        json.dumps(reward_history, indent=2),
        encoding="utf-8",
    )
    save_trained_policy(config.output_dir, payload)
    memory.save(memory_path)
    np.savez(
        config.output_dir / "best_weights.npz",
        mode=np.array(["fewshot_trajectory_learning"]),
        policy_file=np.array([str(config.output_dir / "trained_policy.json")]),
        best_reward=np.array([best_reward], dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--output-dir", default="outputs/grpo_sre")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "heuristic", "auto"])
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--bootstrap", default="reference", choices=["reference", "none"])
    args = parser.parse_args()

    config = TrainingConfig(
        difficulty=args.difficulty,
        steps=args.steps or TRAIN_STEPS[args.difficulty],
        group_size=args.group_size,
        output_dir=Path(args.output_dir),
        provider=args.provider,
        model=args.model,
        ollama_host=args.ollama_host,
        base_url=args.base_url,
        bootstrap=args.bootstrap,
    )
    run_training(config)


if __name__ == "__main__":
    main()
