"""Typed Pydantic models for the SRE incident-response environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field

ServiceName = Literal["api-gateway", "cache", "database", "worker"]
ServiceStatus = Literal["healthy", "degraded", "overloaded", "crashed"]
MetricName = Literal["cpu", "memory", "latency", "error_rate", "throughput"]
ToolName = Literal[
    "get_logs",
    "get_metrics",
    "get_dependencies",
    "restart",
    "scale",
    "rollback",
    "classify_vuln",
    "apply_patch",
    "verify_patch",
    "post_mortem",
]
Difficulty = Literal["easy", "medium", "hard"]
Version = Literal["previous", "stable"]


class SREAction(Action):
    """One structured command the agent sends per step."""

    model_config = ConfigDict(extra="forbid")

    tool: ToolName = Field(description="Which tool to use.")
    service: ServiceName | None = Field(
        default=None,
        description="Required for service-targeting tools such as restart/get_logs.",
    )
    metric: MetricName | None = Field(
        default=None,
        description="Required for get_metrics. Which metric to query.",
    )
    replicas: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Required for scale. Target replica count (1-5).",
    )
    version: Version | None = Field(
        default=None,
        description="Required for rollback. Which version to roll back to.",
    )
    vulnerability_type: str | None = Field(
        default=None,
        description="For classify_vuln. Example: 'sql_injection' or 'xss'.",
    )
    patch_id: str | None = Field(
        default=None,
        description="For apply_patch. Example: 'parameterized_query'.",
    )
    root_cause: str | None = Field(
        default=None,
        description="For post_mortem. Human-readable root cause summary.",
    )
    attack_vector: str | None = Field(
        default=None,
        description="For post_mortem. Human-readable attack vector summary.",
    )
    fix_sequence: list[str] | None = Field(
        default=None,
        description="For post_mortem. Ordered list of remediation steps taken.",
    )
    prevention: str | None = Field(
        default=None,
        description="For post_mortem. Prevention or follow-up actions.",
    )


class ServiceInfo(BaseModel):
    """Runtime status for a single service."""

    model_config = ConfigDict(extra="forbid")

    name: ServiceName
    status: ServiceStatus
    cpu_pct: float = Field(ge=0.0, le=100.0)
    memory_pct: float = Field(ge=0.0, le=100.0)
    error_rate_pct: float = Field(ge=0.0, le=100.0)
    latency_ms: float = Field(ge=0.0)
    replicas: int = Field(default=1, ge=1, le=5)


class Alert(BaseModel):
    """Alert surfaced to the agent."""

    model_config = ConfigDict(extra="forbid")

    service: ServiceName
    severity: Literal["info", "warning", "critical"]
    message: str
    active: bool = True


class SREObservation(Observation):
    """Everything the agent receives after each action."""

    model_config = ConfigDict(extra="forbid")

    tick: int = Field(description="Current step. Max = 20.")
    max_ticks: int = Field(default=20)
    difficulty: Difficulty
    services: dict[str, ServiceInfo] = Field(
        description="Current status of all 4 services."
    )
    active_alerts: list[Alert] = Field(
        description="All currently firing alerts. Empty means system healthy."
    )
    last_action_result: str = Field(
        default="",
        description="Human-readable result of the previous action.",
    )
    tool_output: str | None = Field(
        default=None,
        description="Data returned by investigation tools. None if last action was a fix.",
    )
    episode_complete: bool = Field(default=False)
    success: bool = Field(default=False)
    failure_reason: str | None = Field(default=None)
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    phase: str = Field(
        default="Phase 1: Infrastructure Triage",
        description="Current phase label for the unified environment.",
    )
    phase2_unlocked: bool = Field(
        default=False,
        description="True once the security indicator has been discovered in logs.",
    )
    phase2_complete: bool = Field(
        default=False,
        description="True once the security fix has been correctly verified.",
    )
    security_sub_quest: dict[str, Any] | None = Field(
        default=None,
        description="Security task data revealed after the unlock condition is met.",
    )
    websec_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Security-subquest progress such as classification or patch status.",
    )
    postmortem_submitted: bool = Field(
        default=False,
        description="True once a post_mortem action has been submitted.",
    )
    postmortem_available: bool = Field(
        default=False,
        description="True from tick 5 onward in the unified environment.",
    )
    final_score: float = Field(
        default=0.0,
        description="Normalized 0.0-1.0 episode score once grading is available.",
    )
    phase_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-phase score breakdown for infrastructure, security, and postmortem.",
    )


class SREState(State):
    """Episode-level metadata returned by ``state``."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    step_count: int
    difficulty: Difficulty
    scenario_id: str
    cumulative_reward: float
    current_tick: int
    max_ticks: int
    investigated_root_cause_service: bool
    fix_attempts: int
    wrong_fix_attempts: int
    correct_fixes_applied: int
    all_services_healthy: bool
    episode_complete: bool
    phase2_unlocked: bool = False
    phase2_complete: bool = False
    postmortem_submitted: bool = False
    postmortem_score: float = 0.0
    final_score: float = 0.0


class ScenarioSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    difficulty: Difficulty
    name: str
    description: str
    root_cause_service: ServiceName
    correct_fix_steps: int


class ScenarioCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: str = "sre_env"
    default_scenario_id: str = "easy_001"
    available_difficulties: list[Difficulty]
    filtered_difficulty: Difficulty | None = None
    scenarios: list[ScenarioSummary]


class GraderCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str
    weight: float


class GraderReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    message: str
    checks: list[GraderCheck]


class BaselineStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    service: ServiceName | None = None
    metric: MetricName | None = None
    replicas: int | None = None
    version: Version | None = None
    rationale: str = ""


class BaselineDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    name: str
    description: str
    optimal_steps: int
    actions: list[BaselineStep]


class BaselineCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: str = "sre_env"
    baselines: list[BaselineDefinition]


class RuntimeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: str = "sre_env"
    progress: SREState
    grader: GraderReport
