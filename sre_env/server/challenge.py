"""Task catalog, baseline trajectories, and grader logic for REST endpoints."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from ..models import (
    BaselineCatalog,
    BaselineDefinition,
    BaselineStep,
    GraderCheck,
    GraderReport,
    ScenarioCatalog,
    ScenarioSummary,
)

SCENARIOS_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"
UNIFIED_SCENARIOS_PATH = Path(__file__).parent.parent / "data" / "unified_scenarios.json"
DEFAULT_SCENARIO_ID = "easy_001"


def _load_scenarios() -> dict[str, dict]:
    with SCENARIOS_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {scenario["id"]: scenario for scenario in data["scenarios"]}


SCENARIOS = _load_scenarios()


def _load_unified_scenarios() -> dict[str, dict]:
    try:
        with UNIFIED_SCENARIOS_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    return {scenario["id"]: scenario for scenario in data["scenarios"]}


UNIFIED_SCENARIOS = _load_unified_scenarios()

BASELINES = {
    "easy_001": BaselineDefinition(
        scenario_id="easy_001",
        name="Easy - Database OOM Crash",
        description="Inspect database logs to confirm OOM, then restart database.",
        optimal_steps=2,
        actions=[
            BaselineStep(
                tool="get_logs",
                service="database",
                rationale="Confirm the root cause from logs.",
            ),
            BaselineStep(
                tool="restart",
                service="database",
                rationale="Restart the crashed database process.",
            ),
        ],
    ),
    "medium_001": BaselineDefinition(
        scenario_id="medium_001",
        name="Medium - Cache Failure Cascade",
        description=(
            "Inspect cache and gateway evidence, restart cache first, then restart "
            "database after load recovers."
        ),
        optimal_steps=4,
        actions=[
            BaselineStep(
                tool="get_logs",
                service="cache",
                rationale="Confirm the cache OOM crash.",
            ),
            BaselineStep(
                tool="get_logs",
                service="api-gateway",
                rationale="Confirm all requests are bypassing cache.",
            ),
            BaselineStep(
                tool="restart",
                service="cache",
                rationale="Restore the root cause first.",
            ),
            BaselineStep(
                tool="restart",
                service="database",
                rationale="Clear overload after cache recovers.",
            ),
        ],
    ),
    "hard_001": BaselineDefinition(
        scenario_id="hard_001",
        name="Hard - Bad Deploy Memory Leak Cascade",
        description=(
            "Confirm the worker leak and the resulting database corruption, then "
            "rollback worker, restart database, and restart the gateway."
        ),
        optimal_steps=6,
        actions=[
            BaselineStep(
                tool="get_metrics",
                service="worker",
                metric="memory",
                rationale="Confirm runaway memory growth.",
            ),
            BaselineStep(
                tool="get_logs",
                service="worker",
                rationale="Confirm the bad deployment and connection leak.",
            ),
            BaselineStep(
                tool="get_logs",
                service="database",
                rationale="Confirm the worker is corrupting the pool.",
            ),
            BaselineStep(
                tool="rollback",
                service="worker",
                version="previous",
                rationale="Remove the root cause first.",
            ),
            BaselineStep(
                tool="restart",
                service="database",
                rationale="Reset the corrupted connection pool.",
            ),
            BaselineStep(
                tool="restart",
                service="api-gateway",
                rationale="Clear stale gateway connections.",
            ),
        ],
    ),
}

PHASE2_BASELINES = {
    "easy_001": {
        "scenario_id": "easy_001",
        "name": "Easy - DB Crash (SQLi root cause)",
        "description": "5-tick optimal solve with infrastructure, security patching, and post-mortem.",
        "optimal_tick_count": 5,
        "phases": {
            "phase1_fix_sequence": ["restart database"],
            "phase2_task": "sqli_login",
            "phase2_correct_vuln": "sql_injection",
            "phase2_correct_patch": "parameterized_query",
            "phase3_required_keywords": ["database", "sql injection", "oom"],
        },
    },
    "medium_001": {
        "scenario_id": "medium_001",
        "name": "Medium - Cache Cascade (XSS root cause)",
        "description": "7-tick optimal solve with cache recovery, XSS patching, and post-mortem.",
        "optimal_tick_count": 7,
        "phases": {
            "phase1_fix_sequence": ["restart cache", "restart database"],
            "phase2_task": "xss_comments",
            "phase2_correct_vuln": "xss",
            "phase2_correct_patch": "html_escape",
            "phase3_required_keywords": ["cache", "xss", "session"],
        },
    },
    "hard_001": {
        "scenario_id": "hard_001",
        "name": "Hard - Bad Deploy (Broken Auth root cause)",
        "description": "9-tick optimal solve with deploy rollback, auth fix, and post-mortem.",
        "optimal_tick_count": 9,
        "phases": {
            "phase1_fix_sequence": [
                "rollback worker",
                "restart database",
                "restart api-gateway",
            ],
            "phase2_task": "broken_auth_admin",
            "phase2_correct_vuln": "broken_auth",
            "phase2_correct_patch": "require_admin_role",
            "phase3_required_keywords": ["worker", "broken auth", "deploy"],
        },
    },
}


def _default_progress() -> dict:
    return {
        "episode_id": "bootstrap",
        "step_count": 0,
        "difficulty": "easy",
        "scenario_id": DEFAULT_SCENARIO_ID,
        "cumulative_reward": 0.0,
        "current_tick": 0,
        "max_ticks": 20,
        "investigated_root_cause_service": False,
        "fix_attempts": 0,
        "wrong_fix_attempts": 0,
        "correct_fixes_applied": 0,
        "all_services_healthy": False,
        "episode_complete": False,
    }


_CURRENT_PROGRESS: dict = _default_progress()


def get_scenario(scenario_id: str) -> dict:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario_id {scenario_id!r}. Valid: {valid}") from exc


def list_scenarios(difficulty: str | None = None) -> ScenarioCatalog:
    difficulties = sorted({scenario["difficulty"] for scenario in SCENARIOS.values()})
    if difficulty is not None and difficulty not in difficulties:
        valid = ", ".join(difficulties)
        raise ValueError(f"Unknown difficulty {difficulty!r}. Valid: {valid}")

    scenarios = [
        ScenarioSummary(
            id=scenario["id"],
            difficulty=scenario["difficulty"],
            name=scenario["name"],
            description=scenario["description"],
            root_cause_service=scenario["root_cause_service"],
            correct_fix_steps=len(scenario["correct_fix_sequence"]),
        )
        for scenario in SCENARIOS.values()
        if difficulty is None or scenario["difficulty"] == difficulty
    ]

    return ScenarioCatalog(
        available_difficulties=difficulties,
        filtered_difficulty=difficulty,
        scenarios=scenarios,
    )


def list_baselines(scenario_id: str | None = None) -> BaselineCatalog:
    if scenario_id is None:
        return BaselineCatalog(baselines=list(BASELINES.values()))
    get_scenario(scenario_id)
    return BaselineCatalog(baselines=[BASELINES[scenario_id]])


def set_runtime_progress(state: dict) -> None:
    global _CURRENT_PROGRESS
    _CURRENT_PROGRESS = deepcopy(state)


def current_runtime_progress() -> dict:
    return deepcopy(_CURRENT_PROGRESS)


def grade_episode(state: dict) -> GraderReport:
    """Compute a normalized 0.0-1.0 score for an episode."""

    scenario_id = state.get("scenario_id", DEFAULT_SCENARIO_ID)
    scenario = get_scenario(scenario_id)

    if scenario_id in UNIFIED_SCENARIOS and (
        "phase2_complete" in state
        or "postmortem_submitted" in state
        or "final_score" in state
    ):
        all_healthy = state.get("all_services_healthy", False)
        phase2_complete = state.get("phase2_complete", False)
        postmortem_submitted = state.get("postmortem_submitted", False)
        score = round(float(state.get("final_score", 0.0)), 4)
        passed = all_healthy and phase2_complete and postmortem_submitted
        checks = [
            GraderCheck(
                name="infrastructure_restored",
                passed=all_healthy,
                detail="All 4 services are healthy." if all_healthy else "Infrastructure is still degraded.",
                weight=0.50,
            ),
            GraderCheck(
                name="security_complete",
                passed=phase2_complete,
                detail="Security sub-quest fully resolved." if phase2_complete else "Security remediation is incomplete.",
                weight=0.30,
            ),
            GraderCheck(
                name="postmortem_submitted",
                passed=postmortem_submitted,
                detail="Post-mortem submitted." if postmortem_submitted else "Post-mortem missing.",
                weight=0.20,
            ),
        ]
        return GraderReport(
            scenario_id=scenario_id,
            passed=passed,
            score=max(0.0, min(1.0, score)),
            message="Unified incident fully resolved." if passed else "Unified incident partially resolved.",
            checks=checks,
        )

    investigated = state.get("investigated_root_cause_service", False)
    all_healthy = state.get("all_services_healthy", False)
    ticks = state.get("current_tick", 20)
    max_ticks = state.get("max_ticks", 20)
    wrong_fixes = state.get("wrong_fix_attempts", 0)
    correct_fixes = state.get("correct_fixes_applied", 0)
    total_required = len(scenario["correct_fix_sequence"])

    c1_passed = all_healthy
    c2_passed = investigated
    c3_passed = wrong_fixes == 0
    c4_passed = correct_fixes == total_required
    efficiency = max(0.0, 1.0 - (ticks / max_ticks))

    score = 0.0
    if c1_passed:
        score += 0.40
    if c2_passed:
        score += 0.20
    if c4_passed:
        score += 0.25
    if c3_passed:
        score += 0.05
    score += 0.10 * efficiency
    score -= 0.05 * wrong_fixes
    score = round(max(0.0, min(1.0, score)), 4)

    checks = [
        GraderCheck(
            name="all_services_healthy",
            passed=c1_passed,
            weight=0.40,
            detail=(
                "All 4 services are healthy."
                if c1_passed
                else "Some services are still degraded."
            ),
        ),
        GraderCheck(
            name="investigated_root_cause",
            passed=c2_passed,
            weight=0.20,
            detail=(
                "Agent investigated the root cause service before fixing."
                if c2_passed
                else "Agent did not investigate the root cause service."
            ),
        ),
        GraderCheck(
            name="correct_fix_sequence",
            passed=c4_passed,
            weight=0.25,
            detail=(
                f"All {total_required} correct fixes applied."
                if c4_passed
                else f"Only {correct_fixes}/{total_required} correct fixes applied."
            ),
        ),
        GraderCheck(
            name="no_wrong_fixes",
            passed=c3_passed,
            weight=0.05,
            detail=(
                "No trap actions taken."
                if c3_passed
                else f"{wrong_fixes} wrong fix attempt(s) penalized."
            ),
        ),
        GraderCheck(
            name="efficiency",
            passed=efficiency > 0.5,
            weight=0.10,
            detail=f"Resolved in {ticks}/{max_ticks} ticks (efficiency={efficiency:.2f}).",
        ),
    ]

    passed = all_healthy and correct_fixes == total_required
    return GraderReport(
        scenario_id=scenario_id,
        passed=passed,
        score=score,
        message="Incident fully resolved." if passed else "Incident not fully resolved.",
        checks=checks,
    )


def list_unified_scenarios(difficulty: str | None = None) -> dict:
    scenarios = list(UNIFIED_SCENARIOS.values())
    if difficulty is not None:
        scenarios = [scenario for scenario in scenarios if scenario.get("difficulty") == difficulty]

    return {
        "environment": "sre_env",
        "version": "2.0.0",
        "phases": ["infrastructure", "security", "post_mortem"],
        "tick_budgets": {"easy": 15, "medium": 20, "hard": 25},
        "scenarios": [
            {
                "id": scenario["id"],
                "difficulty": scenario["difficulty"],
                "max_ticks": scenario.get("difficulty_max_ticks", 20),
                "name": scenario["name"],
                "root_cause_service": scenario["root_cause_service"],
                "security_task": scenario["security_sub_quest"]["task_id"],
                "correct_fix_steps": len(scenario["correct_fix_sequence"]),
                "optimal_total_ticks": PHASE2_BASELINES.get(scenario["id"], {}).get(
                    "optimal_tick_count", "?"
                ),
            }
            for scenario in scenarios
        ],
    }


def list_phase2_baselines(scenario_id: str | None = None) -> dict:
    if scenario_id is None:
        baselines = list(PHASE2_BASELINES.values())
    else:
        if scenario_id not in PHASE2_BASELINES:
            valid = ", ".join(sorted(PHASE2_BASELINES))
            raise ValueError(f"Unknown scenario_id {scenario_id!r}. Valid: {valid}")
        baselines = [PHASE2_BASELINES[scenario_id]]
    return {"environment": "sre_env", "baselines": baselines}
