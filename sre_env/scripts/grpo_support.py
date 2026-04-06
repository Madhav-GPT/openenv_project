"""Shared prompting and stored-policy helpers for Ollama-only training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sre_env.models import SREAction

DEFAULT_TRAINED_POLICY_FILE = "trained_policy.json"

SYSTEM_PROMPT = """You are an incident-response agent for the unified SRE-Env benchmark.
Respond with JSON only.
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

Investigate before fixing. Security actions unlock after a SECURITY_ALERT is discovered.
When post_mortem is available, submit a concise structured incident analysis."""


def compact_action(action: SREAction) -> dict[str, Any]:
    data = action.model_dump(exclude_none=True)
    if data.get("metadata") == {}:
        data.pop("metadata", None)
    return data


def parse_action_output(raw: str, fallback: SREAction | None = None) -> SREAction | None:
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


def build_policy_prompt(obs: dict[str, Any]) -> str:
    alerts = obs.get("active_alerts", [])
    services = obs.get("services", {})
    lines = [f"TICK {obs['tick']}/{obs['max_ticks']}"]
    lines.append(f"PHASE: {obs.get('phase', 'Phase 1')}")
    lines.append("ALERTS:")
    if alerts:
        for alert in alerts:
            lines.append(
                f"- [{alert['severity'].upper()}] {alert['service']}: {alert['message']}"
            )
    else:
        lines.append("- none")
    lines.append("SERVICES:")
    for name, service in services.items():
        lines.append(
            f"- {name}: {service['status']} cpu={service['cpu_pct']} "
            f"err={service['error_rate_pct']} mem={service['memory_pct']}"
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


def summarize_actions(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "No stored trajectory yet."
    return "\n".join(
        f"{idx + 1}. {json.dumps(action, separators=(',', ':'), ensure_ascii=True)}"
        for idx, action in enumerate(actions)
    )


def resolve_trained_policy_path(weights_path: str) -> Path:
    path = Path(weights_path)
    if path.is_dir():
        policy_path = path / DEFAULT_TRAINED_POLICY_FILE
    elif path.name == DEFAULT_TRAINED_POLICY_FILE:
        policy_path = path
    else:
        policy_path = path.parent / DEFAULT_TRAINED_POLICY_FILE
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing stored policy artifact: {policy_path}")
    return policy_path


def load_trained_policy(weights_path: str) -> dict[str, list[dict[str, Any]]]:
    policy_path = resolve_trained_policy_path(weights_path)
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    return data.get("policies", {})


def save_trained_policy(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / DEFAULT_TRAINED_POLICY_FILE
    policy_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return policy_path
