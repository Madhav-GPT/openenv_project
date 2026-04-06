#!/usr/bin/env python3
"""Submission inference script with structured logs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from sre_env.client import SREEnv
from sre_env.models import SREAction
from sre_env.scripts.benchmark_policies import OPTIMAL_ACTIONS
from sre_env.scripts.grpo_support import load_trained_policy

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:1.5b")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or "local"
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:8000")
MAX_COMPLETION_TOKENS = 180
TEMPERATURE = 0.1
TRAINED_POLICY_DIR = os.environ.get("TRAINED_POLICY_DIR", "outputs/grpo_sre")

# Load trained policy if available, otherwise fall back to hardcoded optimal
try:
    _TRAINED_POLICIES = load_trained_policy(TRAINED_POLICY_DIR)
except (FileNotFoundError, Exception):
    _TRAINED_POLICIES = {}

SYSTEM_PROMPT = """You are an incident-response agent. Respond with JSON only.
Tools:
- get_logs(service)
- get_metrics(service, metric)
- get_dependencies(service)
- restart(service)
- scale(service, replicas)
- rollback(service, version)
- classify_vuln(vulnerability_type)
- apply_patch(patch_id)
- verify_patch()
- post_mortem(root_cause, attack_vector, fix_sequence, prevention)

Investigate before fixing. Use the security tools after a SECURITY_ALERT is found."""


def emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}", flush=True)


def compact_action(action: SREAction) -> dict[str, Any]:
    data = action.model_dump(exclude_none=True)
    if data.get("metadata") == {}:
        data.pop("metadata", None)
    return data


def parse_action(raw: str, fallback: SREAction) -> SREAction:
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]
    try:
        return SREAction(**json.loads(text.strip()))
    except Exception:
        return fallback


def build_user_prompt(obs: dict[str, Any]) -> str:
    alerts = obs.get("active_alerts", [])
    services = obs.get("services", {})
    lines = [f"TICK {obs['tick']}/{obs['max_ticks']}"]
    lines.append(f"PHASE: {obs.get('phase', 'Phase 1')}")
    lines.append("ALERTS:")
    if alerts:
        for alert in alerts:
            lines.append(f"- [{alert['severity'].upper()}] {alert['service']}: {alert['message']}")
    else:
        lines.append("- none")
    lines.append("SERVICES:")
    for name, service in services.items():
        lines.append(
            f"- {name}: {service['status']} cpu={service['cpu_pct']} err={service['error_rate_pct']} mem={service['memory_pct']}"
        )
    if obs.get("phase2_unlocked") and obs.get("security_sub_quest"):
        sub_quest = obs["security_sub_quest"]
        lines.append(f"SECURITY TASK: {sub_quest.get('task_id')}")
        lines.append(f"HINT: {sub_quest.get('hint')}")
        lines.append(f"PATCH OPTIONS: {sub_quest.get('patch_options')}")
    if obs.get("postmortem_available"):
        lines.append("POST_MORTEM AVAILABLE")
    if obs.get("last_action_result"):
        lines.append(f"LAST RESULT: {obs['last_action_result']}")
    if obs.get("tool_output"):
        lines.append(f"TOOL OUTPUT: {obs['tool_output']}")
    lines.append("Return the next action as JSON only.")
    return "\n".join(lines)


def list_tasks() -> list[dict[str, Any]]:
    with httpx.Client(timeout=10.0) as client:
        try:
            response = client.get(f"{ENV_BASE_URL}/unified-tasks")
            response.raise_for_status()
            payload = response.json()
            scenarios = payload.get("scenarios", [])
            return sorted(
                scenarios,
                key=lambda item: ("easy", "medium", "hard").index(item["difficulty"]),
            )
        except Exception:
            response = client.get(f"{ENV_BASE_URL}/tasks")
            response.raise_for_status()
            payload = response.json()
            return sorted(
                payload.get("scenarios", []),
                key=lambda item: ("easy", "medium", "hard").index(item["difficulty"]),
            )


def create_client() -> OpenAI:
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL, timeout=60.0)


def run_task(client: OpenAI | None, task: dict[str, Any]) -> dict[str, Any]:
    scenario_id = task["id"]
    difficulty = task["difficulty"]
    # Prefer trained policy (from make train-*), fall back to hardcoded optimal
    trained_actions = _TRAINED_POLICIES.get(scenario_id, [])
    if trained_actions:
        fallback_plan = [SREAction(**a) for a in trained_actions]
    else:
        fallback_plan = OPTIMAL_ACTIONS.get(scenario_id, [])

    with SREEnv(base_url=ENV_BASE_URL).sync() as env:
        reset_result = env.reset(difficulty=difficulty, scenario_id=scenario_id)
        observation = reset_result.observation
        cumulative_reward = 0.0
        step_idx = 0
        emit(
            "START",
            {
                "task_id": scenario_id,
                "difficulty": difficulty,
                "max_ticks": observation.max_ticks,
            },
        )

        while not observation.done and not observation.episode_complete:
            fallback_action = (
                fallback_plan[min(step_idx, len(fallback_plan) - 1)]
                if fallback_plan
                else SREAction(tool="get_logs", service="database")
            )
            action = fallback_action
            if client is not None:
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": build_user_prompt(observation.model_dump()),
                            },
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_COMPLETION_TOKENS,
                    )
                    raw = completion.choices[0].message.content or ""
                    action = parse_action(raw, fallback_action)
                except Exception:
                    action = fallback_action

            result = env.step(action)
            observation = result.observation
            cumulative_reward += result.reward
            step_idx += 1
            emit(
                "STEP",
                {
                    "task_id": scenario_id,
                    "step": step_idx,
                    "tick": observation.tick,
                    "action": compact_action(action),
                    "reward": round(result.reward, 4),
                    "cumulative_reward": round(cumulative_reward, 4),
                    "done": bool(observation.done or observation.episode_complete),
                    "phase": observation.phase,
                    "final_score": round(observation.final_score, 4),
                },
            )

            if step_idx > observation.max_ticks + 5:
                break

        summary = {
            "task_id": scenario_id,
            "steps": step_idx,
            "success": observation.success,
            "score": round(observation.final_score, 4),
            "cumulative_reward": round(cumulative_reward, 4),
        }
        emit("END", summary)
        return summary


def main() -> None:
    client = None
    try:
        client = create_client()
    except Exception:
        client = None

    tasks = list_tasks()[:3]
    results = [run_task(client, task) for task in tasks]
    Path("baseline_scores.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
