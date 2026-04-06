"""FastAPI app with OpenEnv routes plus debug endpoints."""

from __future__ import annotations

import argparse
from typing import Any

from fastapi import HTTPException
from openenv.core.env_server import create_app

from ..models import (
    BaselineCatalog,
    GraderReport,
    RuntimeStatus,
    ScenarioCatalog,
    SREAction,
    SREObservation,
    SREState,
)
from .challenge import (
    current_runtime_progress,
    grade_episode,
    list_baselines,
    list_phase2_baselines,
    list_scenarios,
    list_unified_scenarios,
    set_runtime_progress,
)
from .unified_environment import UnifiedSREEnvironment

_BOOTSTRAP_ENV = UnifiedSREEnvironment()
set_runtime_progress(_BOOTSTRAP_ENV.state.model_dump())
app = create_app(
    lambda: UnifiedSREEnvironment(),
    SREAction,
    SREObservation,
    env_name="sre_env",
    max_concurrent_envs=1,
)
app.title = "SRE-Env Incident Response Commander"
app.version = "2.0.0"
app.router.routes = [
    route
    for route in app.router.routes
    if not (getattr(route, "path", None) == "/health")
]


@app.get("/tasks", response_model=ScenarioCatalog, tags=["challenge"])
def tasks(difficulty: str | None = None) -> ScenarioCatalog:
    try:
        return list_scenarios(difficulty=difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/grader", response_model=GraderReport, tags=["challenge"])
def grader(scenario_id: str | None = None) -> GraderReport:
    progress = current_runtime_progress()
    if scenario_id is not None:
        progress["scenario_id"] = scenario_id
    try:
        return grade_episode(progress)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/baseline", response_model=BaselineCatalog, tags=["challenge"])
def baseline(scenario_id: str | None = None) -> BaselineCatalog:
    try:
        return list_baselines(scenario_id=scenario_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/status", response_model=RuntimeStatus, tags=["challenge"])
def status() -> RuntimeStatus:
    progress = current_runtime_progress()
    return RuntimeStatus(
        progress=SREState(**progress),
        grader=grade_episode(progress),
    )


@app.get("/unified-tasks", tags=["phase2"])
def unified_tasks(difficulty: str | None = None) -> dict[str, Any]:
    try:
        return list_unified_scenarios(difficulty=difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/phase2-baseline", tags=["phase2"])
def phase2_baseline(scenario_id: str | None = None) -> dict[str, Any]:
    try:
        return list_phase2_baselines(scenario_id=scenario_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/health", tags=["challenge"])
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "environment": "sre_env",
        "version": "2.0.0",
        "phases": ["infrastructure", "security", "post_mortem"],
        "difficulties": {"easy": 15, "medium": 20, "hard": 25},
    }


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
