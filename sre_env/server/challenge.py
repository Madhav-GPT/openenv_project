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
DEFAULT_SCENARIO_ID = "easy_001"


def _load_scenarios() -> dict[str, dict]:
    with SCENARIOS_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {scenario["id"]: scenario for scenario in data["scenarios"]}


SCENARIOS = _load_scenarios()

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
