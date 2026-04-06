#!/usr/bin/env python3
"""Baseline agent for the SRE environment.

Default behavior is ``auto``:
- Prefer local Ollama with ``qwen2.5:7b`` if available.
- Fall back to a deterministic heuristic policy otherwise.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from sre_env.client import SREEnv
from sre_env.models import SREAction
from sre_env.scripts.benchmark_policies import OPTIMAL_ACTIONS
from sre_env.scripts.live_dashboard import BaselineDashboard
from sre_env.scripts.llm_backends import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    LLMCallResult,
    LLMConfig,
    build_llm_config,
    call_action_model,
)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer responding to a live production incident.

SERVICES: api-gateway, cache, database, worker
DEPENDENCY ORDER: internet -> api-gateway -> cache -> database -> worker

AVAILABLE TOOLS (respond with JSON only):
  {"tool": "get_logs", "service": "<name>"}
  {"tool": "get_metrics", "service": "<name>", "metric": "<cpu|memory|latency|error_rate|throughput>"}
  {"tool": "get_dependencies", "service": "<name>"}
  {"tool": "restart", "service": "<name>"}
  {"tool": "scale", "service": "<name>", "replicas": <1-5>}
  {"tool": "rollback", "service": "<name>", "version": "previous"}
  {"tool": "classify_vuln", "vulnerability_type": "<sql_injection|xss|broken_auth>"}
  {"tool": "apply_patch", "patch_id": "<scenario-specific patch id>"}
  {"tool": "verify_patch"}
  {"tool": "post_mortem", "root_cause": "...", "attack_vector": "...", "fix_sequence": ["..."], "prevention": "..."}

Investigate before fixing. Fix the root cause first. Respond with valid JSON only.
"""


def _action_key(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("tool"),
        action.get("service"),
        action.get("metric"),
        action.get("replicas"),
        action.get("version"),
        action.get("vulnerability_type"),
        action.get("patch_id"),
    )


def _build_user_message(observation: dict[str, Any]) -> str:
    lines = [f"TICK {observation['tick']}/{observation['max_ticks']}"]
    lines.append(f"PHASE: {observation.get('phase', 'Phase 1')}")
    lines.append("ACTIVE ALERTS:")
    alerts = observation.get("active_alerts", [])
    if alerts:
        for alert in alerts:
            lines.append(
                f"- [{alert['severity'].upper()}] {alert['service']}: {alert['message']}"
            )
    else:
        lines.append("- none")

    lines.append("SERVICES:")
    for name, service in observation["services"].items():
        lines.append(
            f"- {name}: {service['status']} cpu={service['cpu_pct']} mem={service['memory_pct']} err={service['error_rate_pct']}"
        )
    if observation.get("last_action_result"):
        lines.append(f"LAST RESULT: {observation['last_action_result']}")
    if observation.get("tool_output"):
        lines.append(f"TOOL OUTPUT: {observation['tool_output']}")
    if observation.get("security_sub_quest"):
        lines.append(
            f"SECURITY SUB-QUEST: {observation['security_sub_quest'].get('task_id', 'unknown')}"
        )
    if observation.get("postmortem_available"):
        lines.append("POST_MORTEM AVAILABLE")
    lines.append("What is the next action? JSON only.")
    return "\n".join(lines)


def _heuristic_action(
    observation: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    seen = {_action_key(action) for action in history}
    difficulty = observation["difficulty"]
    scenario_id = {"easy": "easy_001", "medium": "medium_001", "hard": "hard_001"}[
        difficulty
    ]
    sequence = [
        action.model_dump(exclude_none=True) for action in OPTIMAL_ACTIONS[scenario_id]
    ]

    for action in sequence:
        if _action_key(action) not in seen:
            return action
    return sequence[-1]


def run_episode(
    difficulty: str,
    base_url: str,
    llm_config: LLMConfig,
    dashboard: BaselineDashboard,
) -> float:
    history: list[dict[str, Any]] = []
    llm_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0

    dashboard.start_episode(difficulty)
    with SREEnv(base_url=base_url).sync() as env:
        reset_result = env.reset(difficulty=difficulty)
        observation = reset_result.observation
        done = reset_result.done
        while not done:
            obs_dict = observation.model_dump()
            fallback = _heuristic_action(obs_dict, history)
            llm_history.append(
                {"role": "user", "content": _build_user_message(obs_dict)}
            )
            call_result: LLMCallResult = call_action_model(
                llm_config,
                llm_history,
                fallback,
                temperature=0.1,
                max_tokens=150,
            )
            try:
                action = SREAction(**call_result.action)
            except Exception as exc:
                call_result.metrics.mark_fallback(f"schema validation: {exc}")
                action = SREAction(**fallback)
            history.append(action.model_dump(exclude_none=True))
            llm_history.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action.model_dump(exclude_none=True)),
                }
            )
            result = env.step(action)
            observation = result.observation
            total_reward += result.reward
            done = result.done
            dashboard.record_step(
                difficulty=difficulty,
                action=json.dumps(action.model_dump(exclude_none=True)),
                observation=observation.model_dump(),
                reward=result.reward,
                total_reward=total_reward,
                metrics=call_result.metrics,
            )
        dashboard.finish_episode(
            difficulty=difficulty,
            total_reward=total_reward,
            success=observation.success,
            failure_reason=observation.failure_reason,
        )
        return total_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=SREEnv.DEFAULT_BASE_URL)
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "ollama", "heuristic"],
    )
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    args = parser.parse_args()

    llm_config = build_llm_config(
        provider=args.provider,
        model=args.model,
        ollama_host=args.ollama_host,
    )
    with BaselineDashboard(
        provider=llm_config.provider,
        model=llm_config.model,
    ) as dashboard:
        results = {
            difficulty: run_episode(difficulty, args.base_url, llm_config, dashboard)
            for difficulty in ("easy", "medium", "hard")
        }
        dashboard.log("baseline run complete")

    print("\n=== BASELINE SUMMARY ===")
    print(
        f"provider={llm_config.provider} model={llm_config.model} "
        f"base_url={args.base_url}"
    )
    for difficulty, reward in results.items():
        print(f"{difficulty:>6}: reward={reward:+.2f}")


if __name__ == "__main__":
    main()
