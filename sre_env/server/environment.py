"""Core SRE state machine."""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

from openenv.core.env_server import Environment

from ..models import Alert, SREAction, SREObservation, SREState, ServiceInfo
from .challenge import set_runtime_progress
from .grader import SREGrader

SCENARIOS_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"
MAX_TICKS = 20


def _load_scenarios() -> dict[str, dict]:
    with SCENARIOS_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {scenario["id"]: scenario for scenario in data["scenarios"]}


SCENARIOS = _load_scenarios()
INVESTIGATION_TOOLS = {"get_logs", "get_metrics", "get_dependencies"}


class SREEnvironment(Environment[SREAction, SREObservation, SREState]):
    """One episode equals one incident scenario."""

    def __init__(self) -> None:
        super().__init__()
        self._grader = SREGrader()
        self._ep: dict[str, Any] = {}
        self._init_blank_episode()

    def _init_blank_episode(self) -> None:
        self._ep = self._make_episode(SCENARIOS["easy_001"])
        set_runtime_progress(self._state_dict())

    def _make_episode(self, scenario: dict, episode_id: str | None = None) -> dict:
        services = {
            name: ServiceInfo(name=name, **data)
            for name, data in scenario["initial_state"].items()
        }
        alerts = [Alert(**alert) for alert in scenario["initial_alerts"]]
        return {
            "episode_id": episode_id or str(uuid.uuid4()),
            "scenario": scenario,
            "difficulty": scenario["difficulty"],
            "tick": 0,
            "max_ticks": MAX_TICKS,
            "services": services,
            "alerts": alerts,
            "cumulative_reward": 0.0,
            "investigated_root_cause": False,
            "fix_attempts": 0,
            "wrong_fix_attempts": 0,
            "correct_fixes_applied": 0,
            "fix_sequence_progress": 0,
            "episode_complete": False,
            "success": False,
            "failure_reason": None,
        }

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> SREObservation:
        difficulty = kwargs.get("difficulty", "easy")
        scenario_id = kwargs.get("scenario_id")

        if scenario_id is not None:
            scenario = SCENARIOS[scenario_id]
        else:
            rng = random.Random(seed)
            candidates = [
                scenario
                for scenario in SCENARIOS.values()
                if scenario["difficulty"] == difficulty
            ]
            scenario = rng.choice(candidates)

        self._ep = self._make_episode(scenario, episode_id)
        set_runtime_progress(self._state_dict())
        return self._build_obs(
            last_action_result="Incident detected. Investigate and restore all services.",
            tool_output=None,
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: SREAction | dict[str, Any],
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SREObservation:
        del timeout_s, kwargs

        if isinstance(action, dict):
            action = SREAction(**action)

        ep = self._ep
        if ep["episode_complete"]:
            return self._build_obs(
                last_action_result="Episode complete. Call reset() to start a new run.",
                tool_output=None,
                reward=0.0,
                done=True,
            )

        ep["tick"] += 1
        previous_alert_count = len([alert for alert in ep["alerts"] if alert.active])
        investigated_root_cause_before = ep["investigated_root_cause"]
        tool_output: str | None = None
        was_correct_fix = False
        was_trap_action = False

        if action.tool in INVESTIGATION_TOOLS:
            tool_output, last_result = self._handle_investigation(action)
        else:
            last_result, was_correct_fix, was_trap_action = self._handle_fix(action)

        if action.tool in INVESTIGATION_TOOLS:
            if action.service == ep["scenario"]["root_cause_service"]:
                ep["investigated_root_cause"] = True

        current_alert_count = len([alert for alert in ep["alerts"] if alert.active])
        alerts_cleared = max(0, previous_alert_count - current_alert_count)
        all_healthy = self._all_healthy()

        reward = self._grader.compute(
            action_tool=action.tool,
            action_service=action.service,
            root_cause_service=ep["scenario"]["root_cause_service"],
            investigated_root_cause_before=investigated_root_cause_before,
            was_correct_fix=was_correct_fix,
            was_trap_action=was_trap_action,
            alerts_cleared=alerts_cleared,
            all_healthy=all_healthy,
        )
        ep["cumulative_reward"] += reward

        done = False
        if all_healthy:
            ep["success"] = True
            ep["episode_complete"] = True
            done = True
            last_result = f"{last_result} ALL SERVICES HEALTHY - incident resolved.".strip()
        elif ep["tick"] >= ep["max_ticks"]:
            ep["episode_complete"] = True
            ep["failure_reason"] = f"Timeout: {ep['max_ticks']} ticks used."
            done = True
            last_result = f"{last_result} TIME LIMIT REACHED.".strip()

        set_runtime_progress(self._state_dict())
        return self._build_obs(
            last_action_result=last_result,
            tool_output=tool_output,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> SREState:
        return SREState(**self._state_dict())

    def _handle_investigation(self, action: SREAction) -> tuple[str, str]:
        scenario = self._ep["scenario"]
        service = action.service

        if action.tool == "get_logs":
            return (
                scenario["logs"].get(service, f"No logs available for {service}."),
                f"[get_logs] Retrieved logs for {service}.",
            )

        if action.tool == "get_metrics":
            if action.metric is None:
                return (
                    "ERROR: metric parameter required for get_metrics.",
                    "ERROR: metric required.",
                )
            metrics = scenario["metrics"].get(service, {})
            return (
                metrics.get(action.metric, f"No metric {action.metric!r} for {service}."),
                f"[get_metrics] Retrieved {action.metric} for {service}.",
            )

        if action.tool == "get_dependencies":
            dependencies = scenario["dependencies"]
            return (
                dependencies.get(service, f"No dependency info for {service}."),
                f"[get_dependencies] Retrieved dependency info for {service}.",
            )

        return "Unknown tool.", "Unknown tool."

    def _handle_fix(self, action: SREAction) -> tuple[str, bool, bool]:
        ep = self._ep
        scenario = ep["scenario"]
        ep["fix_attempts"] += 1

        progress = ep["fix_sequence_progress"]
        correct_sequence = scenario["correct_fix_sequence"]
        if progress < len(correct_sequence):
            expected = correct_sequence[progress]
            if self._action_matches(expected, action):
                ep["fix_sequence_progress"] += 1
                ep["correct_fixes_applied"] += 1
                message = self._apply_resolution(progress)
                return message, True, False

        for trap in scenario.get("trap_actions", []):
            if self._action_matches(trap["action"], action):
                ep["wrong_fix_attempts"] += 1
                return trap["result"], False, True

        return (
            f"{action.tool} on {action.service}: no effect on the current incident. "
            "System state unchanged.",
            False,
            False,
        )

    def _action_matches(self, expected: dict, action: SREAction) -> bool:
        for field in ("tool", "service", "metric", "replicas", "version"):
            if field in expected and getattr(action, field) != expected[field]:
                return False
        return True

    def _apply_resolution(self, step_idx: int) -> str:
        ep = self._ep
        scenario = ep["scenario"]
        resolution_steps = list(scenario.get("resolution", {}).values())
        if not resolution_steps:
            return "Fix step applied."

        resolution = resolution_steps[min(step_idx, len(resolution_steps) - 1)]
        state_key = "final_state" if "final_state" in resolution else "intermediate_state"
        for service_name, values in resolution.get(state_key, {}).items():
            ep["services"][service_name] = ServiceInfo(name=service_name, **values)
        self._refresh_alerts()
        return resolution.get("message", "Fix step applied.")

    def _refresh_alerts(self) -> None:
        ep = self._ep
        remaining = []
        for alert in ep["alerts"]:
            service = ep["services"][alert.service]
            if service.status == "healthy":
                continue
            remaining.append(alert)
        ep["alerts"] = remaining

    def _all_healthy(self) -> bool:
        return all(service.status == "healthy" for service in self._ep["services"].values())

    def _state_dict(self) -> dict:
        ep = self._ep
        return {
            "episode_id": ep["episode_id"],
            "step_count": ep["tick"],
            "difficulty": ep["difficulty"],
            "scenario_id": ep["scenario"]["id"],
            "cumulative_reward": ep["cumulative_reward"],
            "current_tick": ep["tick"],
            "max_ticks": ep["max_ticks"],
            "investigated_root_cause_service": ep["investigated_root_cause"],
            "fix_attempts": ep["fix_attempts"],
            "wrong_fix_attempts": ep["wrong_fix_attempts"],
            "correct_fixes_applied": ep["correct_fixes_applied"],
            "all_services_healthy": self._all_healthy(),
            "episode_complete": ep["episode_complete"],
        }

    def _build_obs(
        self,
        last_action_result: str,
        tool_output: str | None,
        reward: float,
        done: bool,
    ) -> SREObservation:
        ep = self._ep
        return SREObservation(
            tick=ep["tick"],
            max_ticks=ep["max_ticks"],
            difficulty=ep["difficulty"],
            services=ep["services"],
            active_alerts=[alert for alert in ep["alerts"] if alert.active],
            last_action_result=last_action_result,
            tool_output=tool_output,
            episode_complete=ep["episode_complete"],
            success=ep["success"],
            failure_reason=ep["failure_reason"],
            reward=reward,
            done=done,
        )
