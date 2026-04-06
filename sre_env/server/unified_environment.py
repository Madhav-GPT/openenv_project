"""Unified SRE + WebSec + post-mortem environment."""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..models import Alert, SREAction, SREObservation, SREState, ServiceInfo
from .challenge import set_runtime_progress
from .grader import SREGrader
from .judge import PostMortemJudge
from .websec_grader import WebSecGrader

SCENARIOS_PATH = Path(__file__).parent.parent / "data" / "unified_scenarios.json"
MAX_TICKS_BY_DIFFICULTY = {"easy": 15, "medium": 20, "hard": 25}
POSTMORTEM_MIN_TICK = 5
INVESTIGATION_TOOLS = {"get_logs", "get_metrics", "get_dependencies"}
WEBSEC_TOOLS = {"classify_vuln", "apply_patch", "verify_patch"}
FIX_TOOLS = {"restart", "scale", "rollback"}


def _load_scenarios() -> dict[str, dict]:
    with SCENARIOS_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {scenario["id"]: scenario for scenario in payload["scenarios"]}


SCENARIOS = _load_scenarios()


class UnifiedSREEnvironment(Environment[SREAction, SREObservation, SREState]):
    """Unified three-phase environment used by the HTTP app."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._sre_grader = SREGrader()
        self._websec_grader = WebSecGrader()
        self._judge = PostMortemJudge()
        self._ep: dict[str, Any] = self._make_episode(SCENARIOS["easy_001"])
        set_runtime_progress(self._state_dict())

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="UnifiedSREEnvironment",
            description=(
                "A three-phase incident-response environment combining infrastructure "
                "triage, security remediation, and post-incident reasoning."
            ),
            version="2.0.0",
            author="Madhav",
        )

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
            if not candidates:
                raise ValueError(f"No scenarios found for difficulty '{difficulty}'.")
            scenario = rng.choice(candidates)

        self._ep = self._make_episode(scenario, episode_id)
        set_runtime_progress(self._state_dict())
        return self._build_obs(
            last_action_result=(
                "INCIDENT ACTIVE. Investigate infrastructure, unlock the security clue "
                "from logs, and submit a post-mortem after tick 5."
            ),
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
                "Episode complete. Call reset() to start a new episode.",
                None,
                0.0,
                True,
            )

        if action.tool == "post_mortem":
            if ep["tick"] < POSTMORTEM_MIN_TICK:
                return self._build_obs(
                    f"Post-mortem not yet available. Investigate first (minimum tick {POSTMORTEM_MIN_TICK}).",
                    None,
                    -0.05,
                    False,
                )
            return self._handle_postmortem(action)

        ep["tick"] += 1
        previous_alert_count = len([alert for alert in ep["alerts"] if alert.active])
        investigated_root_cause_before = ep["investigated_root_cause"]
        was_all_healthy_before = self._all_healthy()
        tool_output: str | None = None
        last_result = ""
        reward = 0.0
        was_correct_fix = False
        was_trap_action = False

        if action.tool in INVESTIGATION_TOOLS:
            tool_output, last_result, reward = self._handle_investigation(action)
        elif action.tool in WEBSEC_TOOLS:
            if not ep["phase2_unlocked"]:
                last_result = (
                    "Security sub-quest is not active yet. Inspect the right service logs first."
                )
                reward = -0.05
            else:
                last_result, reward = self._handle_websec(action)
        elif action.tool in FIX_TOOLS:
            last_result, was_correct_fix, was_trap_action = self._handle_fix(action)
            current_alert_count = len([alert for alert in ep["alerts"] if alert.active])
            alerts_cleared = max(0, previous_alert_count - current_alert_count)
            reward = self._sre_grader.compute(
                action_tool=action.tool,
                action_service=action.service,
                root_cause_service=ep["scenario"]["root_cause_service"],
                investigated_root_cause_before=investigated_root_cause_before,
                was_correct_fix=was_correct_fix,
                was_trap_action=was_trap_action,
                alerts_cleared=alerts_cleared,
                all_healthy=self._all_healthy() and not was_all_healthy_before,
            )
        else:
            last_result = f"Unknown tool '{action.tool}'."
            reward = -0.05

        all_healthy = self._all_healthy()
        done = False

        if all_healthy and ep["phase2_complete"] and ep["postmortem_submitted"]:
            ep["success"] = True
            ep["episode_complete"] = True
            reward += 0.50
            done = True
            last_result = (
                f"{last_result} FULL RESOLUTION - infrastructure, security, and post-mortem complete."
            ).strip()
        elif ep["tick"] >= ep["max_ticks"]:
            ep["episode_complete"] = True
            ep["failure_reason"] = f"Timeout: {ep['max_ticks']} ticks used."
            done = True
            last_result = f"{last_result} TIME LIMIT REACHED.".strip()
        elif all_healthy and ep["phase2_complete"] and not ep["postmortem_submitted"]:
            last_result = (
                f"{last_result} Infrastructure and security are resolved. "
                "Submit post_mortem to complete the episode."
            ).strip()
        elif all_healthy and not ep["phase2_complete"]:
            last_result = (
                f"{last_result} Infrastructure restored, but the security issue is still open."
            ).strip()

        ep["cumulative_reward"] += reward
        ep["final_score"] = self._compute_final_score()
        set_runtime_progress(self._state_dict())
        return self._build_obs(last_result, tool_output, reward, done)

    @property
    def state(self) -> SREState:
        return SREState(**self._state_dict())

    def _make_episode(self, scenario: dict, episode_id: str | None = None) -> dict:
        difficulty = scenario["difficulty"]
        max_ticks = scenario.get(
            "difficulty_max_ticks",
            MAX_TICKS_BY_DIFFICULTY.get(difficulty, 20),
        )
        services = {
            name: ServiceInfo(name=name, **data)
            for name, data in scenario["initial_state"].items()
        }
        alerts = [Alert(**alert) for alert in scenario["initial_alerts"]]
        return {
            "episode_id": episode_id or str(uuid.uuid4()),
            "scenario": scenario,
            "difficulty": difficulty,
            "tick": 0,
            "max_ticks": max_ticks,
            "services": services,
            "alerts": alerts,
            "cumulative_reward": 0.0,
            "investigated_root_cause": False,
            "fix_attempts": 0,
            "wrong_fix_attempts": 0,
            "correct_fixes_applied": 0,
            "fix_sequence_progress": 0,
            "phase2_unlocked": False,
            "phase2_complete": False,
            "security_log_found": False,
            "websec_state": {
                "classified": False,
                "vulnerability_type": "",
                "patch_applied": "",
                "exploit_blocked": False,
                "functionality_ok": False,
            },
            "postmortem_submitted": False,
            "postmortem_score": 0.0,
            "postmortem_text": "",
            "episode_complete": False,
            "success": False,
            "failure_reason": None,
            "final_score": 0.0,
        }

    def _handle_investigation(
        self, action: SREAction
    ) -> tuple[str | None, str, float]:
        ep = self._ep
        scenario = ep["scenario"]
        service = action.service or scenario["root_cause_service"]
        reward = -0.05

        if action.tool == "get_logs":
            output = scenario["logs"].get(service, f"No logs available for {service}.")
            result = f"[get_logs] Retrieved logs for {service}."

            indicator_service = scenario.get("security_trigger_log_service")
            indicator_string = scenario.get("security_indicator_string", "SECURITY_ALERT")
            if (
                service == indicator_service
                and indicator_string in output
                and not ep["security_log_found"]
            ):
                ep["security_log_found"] = True
                ep["phase2_unlocked"] = True
                hint = scenario["security_sub_quest"]["hint"]
                output = (
                    f"{output}\n\nSECURITY INDICATOR FOUND: {hint}\n"
                    "Security sub-quest unlocked."
                )
                result = f"[get_logs] Retrieved logs for {service}. Security sub-quest unlocked."
                reward += 0.20

            if service == scenario["root_cause_service"] and not ep["investigated_root_cause"]:
                ep["investigated_root_cause"] = True
                reward += 0.15
            elif service != scenario["root_cause_service"]:
                reward += 0.08
            return output, result, reward

        if action.tool == "get_metrics":
            if action.metric is None:
                return (
                    "ERROR: metric parameter required for get_metrics.",
                    "ERROR: metric parameter missing.",
                    reward,
                )
            metrics = scenario["metrics"].get(service, {})
            output = metrics.get(action.metric, f"No metric '{action.metric}' for {service}.")
            result = f"[get_metrics] Retrieved {action.metric} for {service}."
            if service == scenario["root_cause_service"] and not ep["investigated_root_cause"]:
                ep["investigated_root_cause"] = True
                reward += 0.15
            elif service != scenario["root_cause_service"]:
                reward += 0.08
            return output, result, reward

        if action.tool == "get_dependencies":
            output = scenario["dependencies"].get(
                service, f"No dependency info for {service}."
            )
            return output, f"[get_dependencies] Retrieved dependency info for {service}.", reward + 0.05

        return None, "Unknown investigation tool.", reward

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
                return self._apply_resolution(progress), True, False

        for trap in scenario.get("trap_actions", []):
            if self._action_matches(trap["action"], action):
                ep["wrong_fix_attempts"] += 1
                return trap["result"], False, True

        service = action.service or "unknown"
        return (
            f"{action.tool} on {service}: no observable effect on the current incident.",
            False,
            False,
        )

    def _handle_websec(self, action: SREAction) -> tuple[str, float]:
        ep = self._ep
        ws = ep["websec_state"]
        sub_quest = ep["scenario"]["security_sub_quest"]
        reward = -0.05

        if action.tool == "classify_vuln":
            if not action.vulnerability_type:
                return "vulnerability_type parameter required for classify_vuln.", reward
            if action.vulnerability_type == sub_quest["correct_vulnerability"]:
                ws["classified"] = True
                ws["vulnerability_type"] = action.vulnerability_type
                return (
                    f"Vulnerability correctly classified as '{action.vulnerability_type}'. "
                    "Apply a patch next.",
                    reward + 0.10,
                )
            return (
                f"Incorrect classification '{action.vulnerability_type}'. "
                f"Hint: {sub_quest['scanner_hint']}",
                reward - 0.05,
            )

        if action.tool == "apply_patch":
            if not action.patch_id:
                return "patch_id parameter required for apply_patch.", reward
            if not ws["classified"]:
                return "Classify the vulnerability before applying a patch.", reward
            ws["patch_applied"] = action.patch_id
            ws["exploit_blocked"] = False
            ws["functionality_ok"] = False
            patch_reward = 0.15 if action.patch_id == sub_quest["correct_patch"] else 0.0
            return (
                f"Patch '{action.patch_id}' applied. Run verify_patch to validate it.",
                reward + patch_reward,
            )

        if action.tool == "verify_patch":
            if not ws["patch_applied"]:
                return "Apply a patch before verifying it.", reward
            exploit_blocked, functionality_ok, message = self._websec_grader.verify(
                sub_quest, ws["patch_applied"]
            )
            ws["exploit_blocked"] = exploit_blocked
            ws["functionality_ok"] = functionality_ok
            if exploit_blocked and functionality_ok:
                ep["phase2_complete"] = True
                return (
                    f"{message} {sub_quest['infrastructure_clue']}",
                    reward + 0.25,
                )
            if exploit_blocked and not functionality_ok:
                return (
                    f"{message} Exploit blocked, but application behavior is broken.",
                    reward - 0.05,
                )
            return (
                f"{message} Patch is insufficient - exploit still possible.",
                reward - 0.10,
            )

        return "Unknown WebSec action.", reward

    def _handle_postmortem(self, action: SREAction) -> SREObservation:
        ep = self._ep
        if ep["postmortem_submitted"]:
            return self._build_obs(
                "Post-mortem already submitted for this episode.",
                None,
                0.0,
                ep["episode_complete"],
            )

        ep["postmortem_submitted"] = True
        pm_payload = {
            "root_cause": action.root_cause or "",
            "attack_vector": action.attack_vector or "",
            "fix_sequence": action.fix_sequence or [],
            "prevention": action.prevention or "",
        }
        ep["postmortem_text"] = json.dumps(pm_payload)

        # Fully deterministic post-mortem scoring — no external LLM needed
        ep["postmortem_score"] = self._judge.score(
            postmortem=ep["postmortem_text"],
            scenario=ep["scenario"],
        )

        reward = ep["postmortem_score"]
        ep["cumulative_reward"] += reward

        done = self._all_healthy() and ep["phase2_complete"]
        if done:
            ep["success"] = True
            ep["episode_complete"] = True
        ep["final_score"] = self._compute_final_score()
        set_runtime_progress(self._state_dict())
        return self._build_obs(
            (
                "Post-mortem submitted. "
                f"Score={ep['postmortem_score']:.3f}, "
                f"final score={ep['final_score']:.3f}."
            ),
            None,
            reward,
            done,
        )

    def _apply_resolution(self, step_idx: int) -> str:
        resolution_steps = list(self._ep["scenario"].get("resolution", {}).values())
        if not resolution_steps:
            return "Fix step applied."

        resolution = resolution_steps[min(step_idx, len(resolution_steps) - 1)]
        state_key = "final_state" if "final_state" in resolution else "intermediate_state"
        for service_name, values in resolution.get(state_key, {}).items():
            self._ep["services"][service_name] = ServiceInfo(name=service_name, **values)
        self._refresh_alerts()
        return resolution.get("message", "Fix step applied.")

    def _refresh_alerts(self) -> None:
        self._ep["alerts"] = [
            alert
            for alert in self._ep["alerts"]
            if self._ep["services"][alert.service].status != "healthy"
        ]

    def _action_matches(self, expected: dict[str, Any], action: SREAction) -> bool:
        for field in (
            "tool",
            "service",
            "metric",
            "replicas",
            "version",
            "vulnerability_type",
            "patch_id",
        ):
            if field in expected and getattr(action, field) != expected[field]:
                return False
        return True

    def _all_healthy(self) -> bool:
        return all(service.status == "healthy" for service in self._ep["services"].values())

    def _score_breakdown(self) -> dict[str, float]:
        ep = self._ep
        scenario = ep["scenario"]

        infrastructure = 0.0
        if self._all_healthy():
            infrastructure += 0.25
        if ep["investigated_root_cause"]:
            infrastructure += 0.10
        if ep["correct_fixes_applied"] >= len(scenario["correct_fix_sequence"]):
            infrastructure += 0.10
        efficiency = max(0.0, 1.0 - ep["tick"] / ep["max_ticks"])
        infrastructure += 0.05 * efficiency
        infrastructure -= min(0.20, 0.05 * ep["wrong_fix_attempts"])
        infrastructure = round(max(0.0, min(0.50, infrastructure)), 4)

        security = 0.0
        if ep["phase2_unlocked"]:
            security += 0.05
        if ep["websec_state"].get("classified"):
            security += 0.10
        if ep["websec_state"].get("patch_applied") == scenario["security_sub_quest"]["correct_patch"]:
            security += 0.15
        if ep["websec_state"].get("exploit_blocked"):
            security += 0.05
        if ep["websec_state"].get("functionality_ok"):
            security += 0.05
        security = round(max(0.0, min(0.40, security)), 4)

        postmortem = round(max(0.0, min(0.30, ep["postmortem_score"])), 4)
        return {
            "infrastructure": infrastructure,
            "security": security,
            "postmortem": postmortem,
        }

    def _compute_final_score(self) -> float:
        breakdown = self._score_breakdown()
        return round(min(1.0, sum(breakdown.values())), 4)

    def _state_dict(self) -> dict[str, Any]:
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
            "phase2_unlocked": ep["phase2_unlocked"],
            "phase2_complete": ep["phase2_complete"],
            "postmortem_submitted": ep["postmortem_submitted"],
            "postmortem_score": ep["postmortem_score"],
            "all_services_healthy": self._all_healthy(),
            "episode_complete": ep["episode_complete"],
            "final_score": ep.get("final_score", self._compute_final_score()),
        }

    def _build_obs(
        self,
        last_action_result: str,
        tool_output: str | None,
        reward: float,
        done: bool,
    ) -> SREObservation:
        ep = self._ep
        if ep["episode_complete"]:
            phase = "Complete"
        elif ep["postmortem_submitted"]:
            phase = "Phase 3: Post-Mortem Submitted"
        elif ep["phase2_complete"]:
            phase = "Phase 3: Submit Post-Mortem"
        elif ep["phase2_unlocked"]:
            phase = "Phase 2: Security Sub-Quest ACTIVE"
        else:
            phase = "Phase 1: Infrastructure Triage"

        breakdown = self._score_breakdown()
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
            phase=phase,
            phase2_unlocked=ep["phase2_unlocked"],
            phase2_complete=ep["phase2_complete"],
            security_sub_quest=(
                {
                    "task_id": ep["scenario"]["security_sub_quest"]["task_id"],
                    "hint": ep["scenario"]["security_sub_quest"].get("hint", ""),
                    "scanner_hint": ep["scenario"]["security_sub_quest"].get("scanner_hint", ""),
                    "patch_options": ep["scenario"]["security_sub_quest"].get("patch_options", []),
                }
                if ep["phase2_unlocked"]
                else None
            ),
            websec_state=ep["websec_state"],
            postmortem_submitted=ep["postmortem_submitted"],
            postmortem_available=ep["tick"] >= POSTMORTEM_MIN_TICK,
            final_score=ep.get("final_score", self._compute_final_score()),
            phase_scores=breakdown,
        )
