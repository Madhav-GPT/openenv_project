# 🏆 SRE-Env: Incident Response Commander
## FINAL Antigravity Build Prompt — Complete Implementation

> Archive note: this file is historical build context from an earlier phase.
> The canonical repo surface is now the Ollama-only Phase 2 implementation described in
> [README.md](/Users/madhav_189/Documents/meta_hackathon/madhav_trial/README.md),
> [execution.md](/Users/madhav_189/Documents/meta_hackathon/madhav_trial/execution.md),
> [Makefile](/Users/madhav_189/Documents/meta_hackathon/madhav_trial/Makefile),
> and [inference.py](/Users/madhav_189/Documents/meta_hackathon/madhav_trial/inference.py).

> **How to use:** Paste this entire document as your Project Context in Antigravity.
> Build files IN ORDER — each step depends on the previous one.
> Every `[IMPLEMENT]` section = write that file completely. Every `[CONTEXT]` section = read it, don't skip it.

---

## [CONTEXT] What We Are Building & Why It Wins

A **Site Reliability Engineering simulation environment** for training AI agents via
Reinforcement Learning, built on `meta-pytorch/OpenEnv`. This is NOT a toy — it is a
production-grade simulation where:

- A **4-service dependency graph** means the visibly broken service is often NOT the root cause
- **Trap states** exist: the naive fix (restart the obviously broken thing) makes it worse
- **Dense rewards** incentivize investigation before action — the RL signal is meaningful
- **3 difficulty levels** have fundamentally different multi-step fix sequences
- **REST API extras** (`/tasks`, `/grader`, `/baseline`, `/status`) make it debuggable and demo-ready
- **Learning curve comparison** (matplotlib) visualizes baseline vs. few-shot agent improvement

**Judging criteria this targets:**
1. Does the reward function produce a genuine learning signal? ✅ Yes — investigation rewarded separately from fixing
2. Is the domain non-trivial? ✅ Yes — dependency graph, trap states, sequence-dependent fixes
3. Does it run cleanly? ✅ Yes — Docker + HF Spaces + full test suite
4. Is there visual proof of learning? ✅ Yes — matplotlib learning curve script

**Stack:** FastAPI + WebSocket + Pydantic v2 + Docker + Groq API (Llama-3.1-8b-instant)
**Local machine:** MacBook Pro M4 (Apple Silicon, 16GB RAM) — NO local model inference needed

---

## [CONTEXT] The 4-Service Dependency Graph

```
[ Internet Traffic ]
        ↓
[ api-gateway ]  ← entry point, handles all HTTP traffic
        ↓
[ cache ]        ← Redis-like; api-gateway checks here first on every read
        ↓
[ database ]     ← Postgres-like; used when cache misses or for writes
        ↓
[ worker ]       ← background jobs; polls database queue, writes results back
```

**Critical dependency rules (what makes this hard):**
- If `cache` goes down → `api-gateway` floods `database` with 8x normal load
- If `worker` has a memory leak → it corrupts the `database` connection pool
- Restarting `database` while `cache` is still down = database crashes again immediately
- Restarting `database` while `worker` is still leaking = database corrupts again in 2 minutes

These rules are what create trap states and make the environment genuinely hard.

---

## [CONTEXT] Complete File Structure

```
sre_env/
├── __init__.py
├── models.py                        # Pydantic v2 Action/Observation/State
├── client.py                        # EnvClient subclass
├── openenv.yaml                     # Environment manifest
├── pyproject.toml                   # Package config
├── README.md                        # Environment documentation
├── Makefile                         # Dev workflow shortcuts
├── data/
│   └── scenarios.json               # All fake logs, metrics, alerts, resolutions
├── server/
│   ├── __init__.py
│   ├── app.py                       # FastAPI app + extra REST routes
│   ├── environment.py               # Core state machine
│   ├── grader.py                    # Reward + final score computation
│   ├── challenge.py                 # Task catalog, baseline trajectories, grader API
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   ├── baseline_agent.py            # Groq/Llama-3 agent — solves all 3 difficulties
│   └── learning_curve.py            # Matplotlib: baseline vs few-shot comparison
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_environment.py
    ├── test_grader.py
    └── test_http_routes.py
```

---

## [IMPLEMENT] File 1: sre_env/models.py

```python
# sre_env/models.py
"""Typed Pydantic v2 models for the SRE Incident Response environment."""

from __future__ import annotations
from typing import Literal, Optional, Dict, List
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


# ── Type aliases ──────────────────────────────────────────────────────────────

ServiceName  = Literal["api-gateway", "cache", "database", "worker"]
ServiceStatus = Literal["healthy", "degraded", "overloaded", "crashed"]
MetricName   = Literal["cpu", "memory", "latency", "error_rate", "throughput"]
ToolName     = Literal[
    "get_logs", "get_metrics", "get_dependencies",   # investigation
    "restart", "scale", "rollback"                   # fix
]
Difficulty   = Literal["easy", "medium", "hard"]
Version      = Literal["previous", "stable"]


# ── Action ────────────────────────────────────────────────────────────────────

class SREAction(Action):
    """
    One structured command the agent sends per step.

    INVESTIGATION tools (return information, cost 1 tick):
      {"tool": "get_logs",        "service": "database"}
      {"tool": "get_metrics",     "service": "cache",     "metric": "memory"}
      {"tool": "get_dependencies","service": "api-gateway"}

    FIX tools (change system state, cost 1 tick, may succeed, fail, or make things worse):
      {"tool": "restart",  "service": "cache"}
      {"tool": "scale",    "service": "api-gateway", "replicas": 3}
      {"tool": "rollback", "service": "worker",      "version": "previous"}
    """
    tool: ToolName = Field(description="Which tool to use.")
    service: ServiceName = Field(description="Which service to target.")
    metric: Optional[MetricName] = Field(
        default=None,
        description="Required for get_metrics. Which metric to query."
    )
    replicas: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="Required for scale. Target replica count (1-5)."
    )
    version: Optional[Version] = Field(
        default=None,
        description="Required for rollback. Which version to roll back to."
    )


# ── Sub-models ────────────────────────────────────────────────────────────────

class ServiceInfo(BaseModel):
    name: ServiceName
    status: ServiceStatus
    cpu_pct: float = Field(ge=0.0, le=100.0)
    memory_pct: float = Field(ge=0.0, le=100.0)
    error_rate_pct: float = Field(ge=0.0, le=100.0)
    latency_ms: float = Field(ge=0.0)
    replicas: int = Field(default=1, ge=1, le=5)


class Alert(BaseModel):
    service: ServiceName
    severity: Literal["info", "warning", "critical"]
    message: str
    active: bool = True


# ── Observation ───────────────────────────────────────────────────────────────

class SREObservation(Observation):
    """Everything the agent receives after each action."""
    tick: int = Field(description="Current step. Max = 20.")
    max_ticks: int = Field(default=20)
    difficulty: Difficulty
    services: Dict[str, ServiceInfo] = Field(
        description="Current status of all 4 services."
    )
    active_alerts: List[Alert] = Field(
        description="All currently firing alerts. Empty = system healthy."
    )
    last_action_result: str = Field(
        default="",
        description="Human-readable result of the previous action."
    )
    tool_output: Optional[str] = Field(
        default=None,
        description="Data returned by investigation tools. None if last action was a fix."
    )
    episode_complete: bool = Field(default=False)
    success: bool = Field(default=False)
    failure_reason: Optional[str] = Field(default=None)


# ── State (episode metadata for RL frameworks) ───────────────────────────────

class SREState(State):
    """
    Episode-level metadata. Returned by client.state().
    RL frameworks use this for training progress tracking.
    """
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


# ── REST API models (for /tasks, /grader, /baseline, /status endpoints) ──────

class ScenarioSummary(BaseModel):
    id: str
    difficulty: Difficulty
    name: str
    description: str
    root_cause_service: ServiceName
    correct_fix_steps: int


class ScenarioCatalog(BaseModel):
    environment: str = "sre_env"
    default_scenario_id: str = "easy_001"
    available_difficulties: List[Difficulty]
    filtered_difficulty: Optional[Difficulty] = None
    scenarios: List[ScenarioSummary]


class GraderCheck(BaseModel):
    name: str
    passed: bool
    detail: str
    weight: float


class GraderReport(BaseModel):
    scenario_id: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    message: str
    checks: List[GraderCheck]


class BaselineStep(BaseModel):
    tool: ToolName
    service: ServiceName
    metric: Optional[MetricName] = None
    replicas: Optional[int] = None
    version: Optional[Version] = None
    rationale: str = ""


class BaselineDefinition(BaseModel):
    scenario_id: str
    name: str
    description: str
    optimal_steps: int
    actions: List[BaselineStep]


class BaselineCatalog(BaseModel):
    environment: str = "sre_env"
    baselines: List[BaselineDefinition]


class RuntimeStatus(BaseModel):
    environment: str = "sre_env"
    progress: SREState
    grader: GraderReport
```

---

## [IMPLEMENT] File 2: sre_env/data/scenarios.json

```json
{
  "scenarios": [

    {
      "id": "easy_001",
      "difficulty": "easy",
      "name": "Database OOM Crash",
      "description": "The database process ran out of memory and crashed. api-gateway throws 502s. Fix = restart database.",
      "root_cause_service": "database",
      "correct_fix_sequence": [
        {"tool": "restart", "service": "database"}
      ],
      "trap_actions": [
        {
          "action": {"tool": "restart", "service": "api-gateway"},
          "result": "api-gateway restarted successfully but 502 errors continue. The database is still down — the problem is upstream of api-gateway."
        },
        {
          "action": {"tool": "restart", "service": "cache"},
          "result": "cache restarted successfully but 502 errors continue. The database is still crashed."
        },
        {
          "action": {"tool": "scale", "service": "api-gateway", "replicas": 3},
          "result": "api-gateway scaled to 3 replicas but 502 errors continue at the same rate. The database is still down — scaling the gateway doesn't help when upstream is dead."
        }
      ],
      "initial_state": {
        "api-gateway": {"status": "degraded",  "cpu_pct": 45, "memory_pct": 30, "error_rate_pct": 48, "latency_ms": 2800, "replicas": 1},
        "cache":       {"status": "healthy",   "cpu_pct": 12, "memory_pct": 20, "error_rate_pct": 0,  "latency_ms": 4,    "replicas": 1},
        "database":    {"status": "crashed",   "cpu_pct": 0,  "memory_pct": 0,  "error_rate_pct": 100,"latency_ms": 0,    "replicas": 1},
        "worker":      {"status": "healthy",   "cpu_pct": 8,  "memory_pct": 15, "error_rate_pct": 0,  "latency_ms": 120,  "replicas": 1}
      },
      "initial_alerts": [
        {"service": "api-gateway", "severity": "critical", "message": "HTTP 502 error rate at 48% — upstream connection refused"},
        {"service": "database",    "severity": "critical", "message": "Process not responding — last seen 4 minutes ago"}
      ],
      "logs": {
        "api-gateway": "2025-01-15 03:41:02 ERROR upstream connect error or disconnect/reset before headers. reset reason: connection failure\n2025-01-15 03:41:03 ERROR connect() failed (111: Connection refused) while connecting to upstream database:5432\n2025-01-15 03:41:03 WARN  retry attempt 3/3 to database:5432 — giving up\n2025-01-15 03:41:05 ERROR 502 Bad Gateway [database pool exhausted]\n2025-01-15 03:40:58 INFO  Request GET /api/users — forwarding to database\n2025-01-15 03:40:59 ERROR database:5432 connection timed out after 5000ms",
        "cache":       "2025-01-15 03:41:00 INFO  PING received — PONG sent\n2025-01-15 03:40:55 INFO  SET user:session:4429 TTL=3600\n2025-01-15 03:40:54 INFO  GET product:catalog:page1 HIT\n2025-01-15 03:40:52 INFO  Memory usage: 204MB / 1024MB (20%)\n2025-01-15 03:40:50 INFO  Active connections: 12",
        "database":    "2025-01-15 03:38:01 ERROR OutOfMemoryError: cannot allocate 524288 bytes — Killed\n2025-01-15 03:38:00 WARN  Shared buffers at 98% capacity\n2025-01-15 03:37:58 WARN  Memory pressure detected — 1024MB / 1024MB used\n2025-01-15 03:37:55 INFO  Checkpoint starting: write\n2025-01-15 03:37:50 INFO  Connection from api-gateway:52441\n[process terminated — no further logs]",
        "worker":      "2025-01-15 03:41:00 INFO  Job queue poll — 0 items pending\n2025-01-15 03:40:55 INFO  Heartbeat OK\n2025-01-15 03:40:50 INFO  Job completed: send_email id=9912 in 340ms\n2025-01-15 03:40:45 INFO  Job completed: generate_report id=9911 in 2100ms"
      },
      "metrics": {
        "api-gateway": {
          "cpu":        "cpu=45% (elevated but not critical — processing error responses)",
          "memory":     "memory=30% (1.2GB/4GB — normal)",
          "latency":    "p50=2800ms p95=8200ms p99=timeout — CRITICAL. Latency spike correlates with database outage 4 min ago.",
          "error_rate": "error_rate=48% — 502s. Spike began at 03:38:02 exactly when database went down.",
          "throughput": "throughput=210 req/s (50% of normal — half of traffic being rejected with 502)"
        },
        "cache": {
          "cpu":        "cpu=12% (normal)",
          "memory":     "memory=20% (204MB/1024MB — healthy)",
          "latency":    "p50=4ms p95=8ms — healthy",
          "error_rate": "error_rate=0% — no errors",
          "throughput": "throughput=1800 ops/s (normal)"
        },
        "database": {
          "cpu":        "cpu=0% — process is NOT running",
          "memory":     "memory=0% — process is NOT running (was killed by OOM at 03:38:01)",
          "latency":    "latency=N/A — process is NOT running",
          "error_rate": "error_rate=100% — all connection attempts failing (ECONNREFUSED)",
          "throughput": "throughput=0 queries/s — process NOT running"
        },
        "worker": {
          "cpu":        "cpu=8% (light load — no database jobs to process)",
          "memory":     "memory=15% (600MB/4GB — healthy)",
          "latency":    "job_duration_avg=N/A — no jobs running (database unavailable)",
          "error_rate": "error_rate=0% — idle, not erroring",
          "throughput": "throughput=0 jobs/min (normally 4/min — paused due to database outage)"
        }
      },
      "dependencies": {
        "api-gateway": "depends_on: [cache, database]. Cache-aside pattern: all GETs check cache first, then database. POSTs go direct to database. With database DOWN, 48% of requests fail.",
        "cache":       "depends_on: []. Standalone Redis-compatible store. No downstream dependencies. Currently healthy.",
        "database":    "depends_on: []. Primary persistent store. Process was OOM-killed at 03:38:01. Needs restart to recover.",
        "worker":      "depends_on: [database]. Background job processor. Polls database job queue every 30s. Currently idle because database is unreachable."
      },
      "resolution": {
        "restart_database": {
          "trigger": {"tool": "restart", "service": "database"},
          "success": true,
          "message": "database process restarted. Memory cleared. Process healthy. api-gateway connections recovering — 502 error rate dropping.",
          "final_state": {
            "api-gateway": {"status": "healthy", "cpu_pct": 22, "memory_pct": 30, "error_rate_pct": 0, "latency_ms": 45,  "replicas": 1},
            "cache":       {"status": "healthy", "cpu_pct": 12, "memory_pct": 20, "error_rate_pct": 0, "latency_ms": 4,   "replicas": 1},
            "database":    {"status": "healthy", "cpu_pct": 18, "memory_pct": 35, "error_rate_pct": 0, "latency_ms": 12,  "replicas": 1},
            "worker":      {"status": "healthy", "cpu_pct": 8,  "memory_pct": 15, "error_rate_pct": 0, "latency_ms": 890, "replicas": 1}
          }
        }
      }
    },

    {
      "id": "medium_001",
      "difficulty": "medium",
      "name": "Cache Failure Cascade",
      "description": "Cache crashed (OOM). api-gateway now hits database directly for every request, overloading it. THE TRAP: restarting database alone will just overload it again immediately because cache is still down. Correct fix = restart cache FIRST, then restart database.",
      "root_cause_service": "cache",
      "correct_fix_sequence": [
        {"tool": "restart", "service": "cache"},
        {"tool": "restart", "service": "database"}
      ],
      "trap_actions": [
        {
          "action": {"tool": "restart", "service": "database"},
          "result": "database restarted but overloaded again within 30 seconds. Cache is still down — api-gateway floods database with 8x normal query load. Error rate climbing back to 85%. You need to fix the ROOT CAUSE (cache) first."
        },
        {
          "action": {"tool": "scale", "service": "database", "replicas": 3},
          "result": "database scaled to 3 replicas but still overloaded. With cache down, api-gateway generates 8x normal query volume — horizontal scaling alone cannot absorb this. Restore the cache first."
        },
        {
          "action": {"tool": "scale", "service": "api-gateway", "replicas": 1},
          "result": "api-gateway is already at 1 replica. No change. The problem is not traffic volume — it is that all traffic is bypassing the cache and hitting database directly."
        }
      ],
      "initial_state": {
        "api-gateway": {"status": "degraded",   "cpu_pct": 71, "memory_pct": 45, "error_rate_pct": 35, "latency_ms": 3200, "replicas": 1},
        "cache":       {"status": "crashed",    "cpu_pct": 0,  "memory_pct": 0,  "error_rate_pct": 100,"latency_ms": 0,    "replicas": 1},
        "database":    {"status": "overloaded", "cpu_pct": 94, "memory_pct": 88, "error_rate_pct": 35, "latency_ms": 4200, "replicas": 1},
        "worker":      {"status": "degraded",   "cpu_pct": 20, "memory_pct": 22, "error_rate_pct": 40, "latency_ms": 8900, "replicas": 1}
      },
      "initial_alerts": [
        {"service": "cache",      "severity": "critical", "message": "Service unreachable — health check failing for 8 minutes"},
        {"service": "database",   "severity": "critical", "message": "CPU at 94% — connection queue backed up (1240 waiting)"},
        {"service": "api-gateway","severity": "warning",  "message": "Latency p95 > 3s — degraded but still serving"},
        {"service": "worker",     "severity": "warning",  "message": "Job queue growing — database too slow to process (847 pending)"}
      ],
      "logs": {
        "api-gateway": "2025-01-15 04:12:01 WARN  cache HIT ratio: 0% — cache unreachable, ALL requests falling through to database\n2025-01-15 04:12:00 ERROR cache:6379 connection refused — bypassing cache, querying database directly\n2025-01-15 04:11:58 WARN  database response time 4200ms (threshold: 500ms)\n2025-01-15 04:11:55 INFO  Request GET /api/products — cache MISS (cache DOWN) — querying database\n2025-01-15 04:11:54 INFO  Request GET /api/users — cache MISS (cache DOWN) — querying database\n2025-01-15 04:11:52 WARN  database connection pool: 95/100 connections used",
        "cache":       "2025-01-15 04:04:10 ERROR SIGKILL received — out of memory\n2025-01-15 04:04:08 WARN  Memory at 99% — eviction failing (maxmemory-policy=noeviction)\n2025-01-15 04:04:05 ERROR Cannot allocate memory for new keys — maxmemory reached\n2025-01-15 04:04:00 INFO  Memory: 1023MB / 1024MB used\n[process terminated by OOM killer]",
        "database":    "2025-01-15 04:12:02 WARN  Too many connections: 1240 queued (max: 100 active)\n2025-01-15 04:12:01 WARN  CPU load 94% — query planner severely degraded\n2025-01-15 04:11:59 ERROR query timeout after 5000ms: SELECT * FROM products WHERE category_id=...\n2025-01-15 04:11:55 WARN  Autovacuum worker blocked — CPU utilization too high\n2025-01-15 04:11:50 INFO  Connection from api-gateway:52109 [total active: 340]\n2025-01-15 04:11:48 INFO  Connection from api-gateway:52108 [total active: 339]",
        "worker":      "2025-01-15 04:12:00 WARN  Job queue depth: 847 pending (normal: <50)\n2025-01-15 04:11:55 ERROR Job id=10441 failed — database query timeout after 5000ms\n2025-01-15 04:11:50 WARN  database response latency 8900ms — jobs critically slow\n2025-01-15 04:11:45 INFO  Polling job queue... 847 items waiting"
      },
      "metrics": {
        "cache": {
          "cpu":        "cpu=0% — process not running (OOM killed 8 minutes ago)",
          "memory":     "memory=0% — process not running",
          "latency":    "latency=N/A — process not running",
          "error_rate": "error_rate=100%",
          "throughput": "throughput=0 ops/s — DOWN"
        },
        "database": {
          "cpu":        "cpu=94% — CRITICAL. Sustained high CPU for 7+ minutes.",
          "memory":     "memory=88% (7GB/8GB) — near limit from connection overhead",
          "latency":    "p50=4200ms p95=9800ms p99=timeout — severely degraded",
          "error_rate": "error_rate=35% — query timeouts under extreme load",
          "throughput": "throughput=1240 connections queued — receiving 8x NORMAL LOAD because cache is bypassed"
        },
        "api-gateway": {
          "cpu":        "cpu=71% — elevated (processing all cache fallthrough logic)",
          "memory":     "memory=45% (1.8GB/4GB)",
          "latency":    "p50=3200ms p95=7100ms — degraded, matching database slowdown",
          "error_rate": "error_rate=35% — error rate matches database error rate exactly",
          "throughput": "throughput=420 req/s — full normal load but 0% cache HIT ratio"
        },
        "worker": {
          "cpu":        "cpu=20%",
          "memory":     "memory=22%",
          "latency":    "job_duration_avg=8900ms (normal: 890ms) — 10x slowdown",
          "error_rate": "error_rate=40% — job failures from database timeouts",
          "throughput": "throughput=0.4 jobs/min (normal: 4) — 90% reduction"
        }
      },
      "dependencies": {
        "api-gateway": "depends_on: [cache, database]. Cache-aside: check cache first on every GET, on MISS go to database. With cache DOWN, 100% of GET requests hit database — generating 8x normal database load.",
        "cache":       "depends_on: []. Standalone. OOM killed 8 minutes ago. When down, api-gateway generates 8x normal database load. THIS IS THE ROOT CAUSE.",
        "database":    "depends_on: []. Currently receiving 8x normal query volume because cache is bypassed. Not the root cause — just the victim.",
        "worker":      "depends_on: [database]. Background processor. Severely degraded because database is overloaded."
      },
      "resolution": {
        "step1_restart_cache": {
          "trigger": {"tool": "restart", "service": "cache"},
          "success": true,
          "message": "cache restarted successfully. Service healthy. api-gateway cache HIT ratio recovering (currently 45%, rising). database load dropping as cache absorbs traffic.",
          "intermediate_state": {
            "api-gateway": {"status": "degraded",   "cpu_pct": 45, "memory_pct": 38, "error_rate_pct": 12, "latency_ms": 1200, "replicas": 1},
            "cache":       {"status": "healthy",    "cpu_pct": 15, "memory_pct": 10, "error_rate_pct": 0,  "latency_ms": 3,    "replicas": 1},
            "database":    {"status": "overloaded", "cpu_pct": 75, "memory_pct": 70, "error_rate_pct": 20, "latency_ms": 2100, "replicas": 1},
            "worker":      {"status": "degraded",   "cpu_pct": 20, "memory_pct": 22, "error_rate_pct": 15, "latency_ms": 3200, "replicas": 1}
          }
        },
        "step2_restart_database": {
          "trigger": {"tool": "restart", "service": "database"},
          "success": true,
          "message": "database restarted. Connection queue cleared. All services recovering. System fully healthy.",
          "final_state": {
            "api-gateway": {"status": "healthy", "cpu_pct": 25, "memory_pct": 32, "error_rate_pct": 0, "latency_ms": 55,  "replicas": 1},
            "cache":       {"status": "healthy", "cpu_pct": 15, "memory_pct": 10, "error_rate_pct": 0, "latency_ms": 3,   "replicas": 1},
            "database":    {"status": "healthy", "cpu_pct": 22, "memory_pct": 40, "error_rate_pct": 0, "latency_ms": 14,  "replicas": 1},
            "worker":      {"status": "healthy", "cpu_pct": 18, "memory_pct": 22, "error_rate_pct": 0, "latency_ms": 910, "replicas": 1}
          }
        }
      }
    },

    {
      "id": "hard_001",
      "difficulty": "hard",
      "name": "Bad Deploy — Memory Leak Cascade",
      "description": "A bad deployment of worker (v2.4.1) introduced an unbounded connection pool bug. Over 90 minutes it consumed all memory AND corrupted the database connection pool. api-gateway cascading 503s. THE TRAP: fixing database or api-gateway first does nothing — worker will re-corrupt them. Correct fix sequence = rollback worker → restart database → restart api-gateway.",
      "root_cause_service": "worker",
      "correct_fix_sequence": [
        {"tool": "rollback", "service": "worker",      "version": "previous"},
        {"tool": "restart",  "service": "database"},
        {"tool": "restart",  "service": "api-gateway"}
      ],
      "trap_actions": [
        {
          "action": {"tool": "restart", "service": "database"},
          "result": "database restarted but worker (still running bad v2.4.1) immediately reconnects and begins leaking connections again. Database will be degraded again within 2 minutes. Root cause still present — you must rollback worker FIRST."
        },
        {
          "action": {"tool": "restart", "service": "api-gateway"},
          "result": "api-gateway restarted but 503s continue immediately. Database is still corrupted by worker. api-gateway is the victim, not the cause."
        },
        {
          "action": {"tool": "rollback", "service": "database", "version": "previous"},
          "result": "database has no deployable previous version — it was not recently deployed. Nothing changed. Check which service was actually deployed recently."
        },
        {
          "action": {"tool": "rollback", "service": "api-gateway", "version": "previous"},
          "result": "api-gateway has no recent deployment to roll back — it was not the service deployed. Nothing changed."
        },
        {
          "action": {"tool": "scale", "service": "worker", "replicas": 1},
          "result": "worker is already at 1 replica. Also, scaling does not fix a memory leak bug — it would just create multiple leaking instances."
        }
      ],
      "initial_state": {
        "api-gateway": {"status": "degraded",   "cpu_pct": 52, "memory_pct": 38, "error_rate_pct": 55,  "latency_ms": 5100, "replicas": 1},
        "cache":       {"status": "healthy",    "cpu_pct": 14, "memory_pct": 22, "error_rate_pct": 0,   "latency_ms": 5,    "replicas": 1},
        "database":    {"status": "overloaded", "cpu_pct": 88, "memory_pct": 92, "error_rate_pct": 55,  "latency_ms": 6800, "replicas": 1},
        "worker":      {"status": "degraded",   "cpu_pct": 95, "memory_pct": 97, "error_rate_pct": 80,  "latency_ms": 0,    "replicas": 1}
      },
      "initial_alerts": [
        {"service": "worker",     "severity": "critical", "message": "Memory at 97% — OOM imminent. Deployed 90 minutes ago (v2.4.1). Previous stable version: v2.4.0"},
        {"service": "database",   "severity": "critical", "message": "Connection pool corrupted — intermittent crashes every 90 seconds"},
        {"service": "api-gateway","severity": "critical", "message": "503 error rate 55% — database connection failures"},
        {"service": "worker",     "severity": "warning",  "message": "CPU 95% — almost entirely garbage collection overhead (memory leak)"}
      ],
      "logs": {
        "api-gateway": "2025-01-15 05:30:01 ERROR 503 Service Unavailable — database connection refused\n2025-01-15 05:30:00 WARN  Database failover triggered — no healthy replica available\n2025-01-15 05:29:58 ERROR connection pool exhausted — all 100 connections in error state\n2025-01-15 05:29:55 INFO  Request POST /api/orders — forwarded to database — TIMEOUT\n2025-01-15 05:29:50 INFO  Request GET /api/products — cache HIT [OK]\n2025-01-15 05:29:48 ERROR Request GET /api/user/profile — database error: connection reset by peer",
        "cache":       "2025-01-15 05:30:00 INFO  PING — PONG\n2025-01-15 05:29:55 INFO  HIT ratio: 68% (normal)\n2025-01-15 05:29:50 INFO  Memory: 225MB / 1024MB (22%) — healthy\n2025-01-15 05:29:45 INFO  Active connections: 14 (normal)\n2025-01-15 05:29:40 INFO  Evictions: 0 (healthy)",
        "database":    "2025-01-15 05:30:02 ERROR connection reset: worker-service:44921 sent malformed packet — pool entry corrupted\n2025-01-15 05:30:00 WARN  32/100 connections in error state (corrupted by worker)\n2025-01-15 05:29:58 ERROR connection reset: worker-service:44918 sent malformed packet\n2025-01-15 05:29:55 WARN  Memory pressure from worker connection leak: 7.3GB/8GB\n2025-01-15 05:29:50 INFO  Connection from api-gateway:61221 — TIMEOUT (pool at capacity)\n2025-01-15 05:29:45 ERROR worker-service connection pool leak detected — connections not released after jobs\n2025-01-15 05:28:00 WARN  worker-service holding 68 persistent connections (normal: 5-10)",
        "worker":      "2025-01-15 05:30:01 ERROR java.lang.OutOfMemoryError: Java heap space\n2025-01-15 05:30:00 WARN  Heap: 7.76GB / 8GB (97%)\n2025-01-15 05:29:55 WARN  GC overhead limit exceeded — 98% of CPU time spent in garbage collection\n2025-01-15 05:29:50 WARN  Connection leak: 68 database connections open (not released after jobs complete)\n2025-01-15 05:29:45 ERROR Job id=11201 failed — OutOfMemoryError during processing\n2025-01-15 05:29:00 INFO  [v2.4.1] Connection pool initialized with unbounded size (BUG: was bounded in v2.4.0)\n2025-01-15 04:00:01 INFO  [v2.4.1] worker-service started — NEW DEPLOYMENT 90 MINUTES AGO"
      },
      "metrics": {
        "worker": {
          "cpu":        "cpu=95% — CRITICAL. Almost 100% GC (garbage collection) overhead from memory leak.",
          "memory":     "memory=97% (7.76GB/8GB) — OOM imminent. Growing at +50MB/minute since v2.4.1 deployed 90 min ago.",
          "latency":    "job_duration_avg=N/A — jobs failing before completion due to OOM",
          "error_rate": "error_rate=80% — OOM errors on almost every job",
          "throughput": "throughput=0 jobs/min. DEPLOYED VERSION: v2.4.1 (90 min ago). PREVIOUS STABLE: v2.4.0"
        },
        "database": {
          "cpu":        "cpu=88% — elevated from malformed packet processing from worker",
          "memory":     "memory=92% (7.3GB/8GB) — elevated from worker connection leak (68 unreleased connections)",
          "latency":    "p50=6800ms p95=timeout — severe degradation",
          "error_rate": "error_rate=55% — connection pool corruption from worker v2.4.1",
          "throughput": "throughput=32% of connections corrupted — legitimate connections competing with worker's leaked ones"
        },
        "api-gateway": {
          "cpu":        "cpu=52% — normal-ish",
          "memory":     "memory=38% (1.5GB/4GB) — normal",
          "latency":    "p50=5100ms p95=timeout",
          "error_rate": "error_rate=55% — 100% of errors trace to database connection failures",
          "throughput": "throughput=190 req/s (55% of normal). Cache is working fine (68% HIT) — only DB-dependent requests fail."
        },
        "cache": {
          "cpu":        "cpu=14% — healthy",
          "memory":     "memory=22% — healthy",
          "latency":    "p50=5ms — healthy",
          "error_rate": "error_rate=0% — healthy, not the problem",
          "throughput": "throughput=1650 ops/s — healthy"
        }
      },
      "dependencies": {
        "api-gateway": "depends_on: [cache, database]. Cache is working fine (68% HIT ratio). All database-dependent requests are failing due to corrupted pool.",
        "cache":       "depends_on: []. Healthy. Not the problem.",
        "database":    "depends_on: []. Receiving malformed connection packets from worker v2.4.1. 32% of pool entries corrupt. Needs restart AFTER worker is rolled back.",
        "worker":      "depends_on: [database]. VERSION v2.4.1 HAS UNBOUNDED CONNECTION POOL BUG. Currently holding 68 persistent database connections (normal: 5-10). Memory leak growing. MUST BE ROLLED BACK TO v2.4.0 FIRST."
      },
      "resolution": {
        "step1_rollback_worker": {
          "trigger": {"tool": "rollback", "service": "worker", "version": "previous"},
          "success": true,
          "message": "worker rolled back to v2.4.0. Memory leak stopped. Connection leak stopped. Database no longer receiving malformed packets. Database still needs restart to clear existing corrupted pool entries.",
          "intermediate_state": {
            "api-gateway": {"status": "degraded",   "cpu_pct": 52, "memory_pct": 38, "error_rate_pct": 45, "latency_ms": 4200, "replicas": 1},
            "cache":       {"status": "healthy",    "cpu_pct": 14, "memory_pct": 22, "error_rate_pct": 0,  "latency_ms": 5,    "replicas": 1},
            "database":    {"status": "overloaded", "cpu_pct": 70, "memory_pct": 75, "error_rate_pct": 40, "latency_ms": 3200, "replicas": 1},
            "worker":      {"status": "healthy",    "cpu_pct": 8,  "memory_pct": 15, "error_rate_pct": 0,  "latency_ms": 920,  "replicas": 1}
          }
        },
        "step2_restart_database": {
          "trigger": {"tool": "restart", "service": "database"},
          "success": true,
          "message": "database restarted. Connection pool cleared and healthy. Error rate dropping on api-gateway.",
          "intermediate_state": {
            "api-gateway": {"status": "degraded", "cpu_pct": 40, "memory_pct": 38, "error_rate_pct": 10, "latency_ms": 1200, "replicas": 1},
            "cache":       {"status": "healthy",  "cpu_pct": 14, "memory_pct": 22, "error_rate_pct": 0,  "latency_ms": 5,    "replicas": 1},
            "database":    {"status": "healthy",  "cpu_pct": 20, "memory_pct": 38, "error_rate_pct": 5,  "latency_ms": 800,  "replicas": 1},
            "worker":      {"status": "healthy",  "cpu_pct": 8,  "memory_pct": 15, "error_rate_pct": 0,  "latency_ms": 920,  "replicas": 1}
          }
        },
        "step3_restart_api_gateway": {
          "trigger": {"tool": "restart", "service": "api-gateway"},
          "success": true,
          "message": "api-gateway restarted. All stale connection state cleared. System fully healthy.",
          "final_state": {
            "api-gateway": {"status": "healthy", "cpu_pct": 22, "memory_pct": 30, "error_rate_pct": 0, "latency_ms": 48,  "replicas": 1},
            "cache":       {"status": "healthy", "cpu_pct": 14, "memory_pct": 22, "error_rate_pct": 0, "latency_ms": 5,   "replicas": 1},
            "database":    {"status": "healthy", "cpu_pct": 19, "memory_pct": 38, "error_rate_pct": 0, "latency_ms": 11,  "replicas": 1},
            "worker":      {"status": "healthy", "cpu_pct": 8,  "memory_pct": 15, "error_rate_pct": 0, "latency_ms": 920, "replicas": 1}
          }
        }
      }
    }

  ]
}
```

---

## [IMPLEMENT] File 3: sre_env/server/challenge.py

```python
# sre_env/server/challenge.py
"""Task catalog, baseline trajectories, and grader logic for REST API endpoints."""

from __future__ import annotations
import json
from copy import deepcopy
from pathlib import Path

try:
    from ..models import (
        ScenarioSummary, ScenarioCatalog, GraderCheck, GraderReport,
        BaselineStep, BaselineDefinition, BaselineCatalog, SREState
    )
except ImportError:
    from models import (  # type: ignore
        ScenarioSummary, ScenarioCatalog, GraderCheck, GraderReport,
        BaselineStep, BaselineDefinition, BaselineCatalog, SREState
    )

SCENARIOS_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"

def _load_scenarios() -> dict:
    with open(SCENARIOS_PATH) as f:
        data = json.load(f)
    return {s["id"]: s for s in data["scenarios"]}

SCENARIOS = _load_scenarios()
DEFAULT_SCENARIO_ID = "easy_001"

# Optimal baseline trajectories (what a perfect agent does)
BASELINES = {
    "easy_001": BaselineDefinition(
        scenario_id="easy_001",
        name="Easy — Database OOM Crash",
        description="Inspect database logs to confirm OOM, then restart database.",
        optimal_steps=2,
        actions=[
            BaselineStep(tool="get_logs", service="database", rationale="Confirm root cause from logs"),
            BaselineStep(tool="restart",  service="database", rationale="Restart the crashed process"),
        ]
    ),
    "medium_001": BaselineDefinition(
        scenario_id="medium_001",
        name="Medium — Cache Failure Cascade",
        description="Check cache logs to confirm it is down, check api-gateway logs to confirm bypass, restart cache first then database.",
        optimal_steps=4,
        actions=[
            BaselineStep(tool="get_logs",  service="cache",       rationale="Confirm cache is OOM killed — root cause"),
            BaselineStep(tool="get_logs",  service="api-gateway", rationale="Confirm 100% cache bypass causing database overload"),
            BaselineStep(tool="restart",   service="cache",       rationale="Fix root cause first"),
            BaselineStep(tool="restart",   service="database",    rationale="Clear overloaded connection queue"),
        ]
    ),
    "hard_001": BaselineDefinition(
        scenario_id="hard_001",
        name="Hard — Bad Deploy Memory Leak Cascade",
        description="Check worker metrics for memory growth, get worker logs to confirm v2.4.1 deployment and leak, check database logs for corrupted connections, rollback worker then restart database then restart api-gateway.",
        optimal_steps=5,
        actions=[
            BaselineStep(tool="get_metrics",      service="worker",      metric="memory",   rationale="Memory at 97% growing — leak indicator"),
            BaselineStep(tool="get_logs",          service="worker",                         rationale="Confirm v2.4.1 deployment and unbounded connection pool bug"),
            BaselineStep(tool="get_logs",          service="database",                       rationale="Confirm connection corruption from worker"),
            BaselineStep(tool="rollback",          service="worker",      version="previous",rationale="Roll back bad deployment — fix root cause"),
            BaselineStep(tool="restart",           service="database",                       rationale="Clear corrupted connection pool"),
            BaselineStep(tool="restart",           service="api-gateway",                    rationale="Clear stale connection state"),
        ]
    ),
}

# Shared runtime progress for /status endpoint
_CURRENT_PROGRESS: dict = {}

def get_scenario(scenario_id: str) -> dict:
    try:
        return SCENARIOS[scenario_id]
    except KeyError:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario_id {scenario_id!r}. Valid: {valid}")

def list_scenarios(difficulty: str | None = None) -> ScenarioCatalog:
    all_scenarios = [
        ScenarioSummary(
            id=s["id"], difficulty=s["difficulty"], name=s["name"],
            description=s["description"], root_cause_service=s["root_cause_service"],
            correct_fix_steps=len(s["correct_fix_sequence"])
        )
        for s in SCENARIOS.values()
    ]
    difficulties = sorted({s["difficulty"] for s in SCENARIOS.values()})
    if difficulty is not None and difficulty not in difficulties:
        raise ValueError(f"Unknown difficulty {difficulty!r}. Valid: {', '.join(difficulties)}")
    filtered = all_scenarios if difficulty is None else [s for s in all_scenarios if s.difficulty == difficulty]
    return ScenarioCatalog(
        available_difficulties=difficulties,
        filtered_difficulty=difficulty,
        scenarios=filtered
    )

def list_baselines(scenario_id: str | None = None) -> BaselineCatalog:
    if scenario_id:
        get_scenario(scenario_id)
        return BaselineCatalog(baselines=[BASELINES[scenario_id]])
    return BaselineCatalog(baselines=list(BASELINES.values()))

def set_runtime_progress(state: dict) -> None:
    global _CURRENT_PROGRESS
    _CURRENT_PROGRESS = deepcopy(state)

def current_runtime_progress() -> dict:
    return deepcopy(_CURRENT_PROGRESS)

def grade_episode(state: dict) -> GraderReport:
    """Compute a normalized 0.0–1.0 score for a completed episode."""
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
    if c1_passed: score += 0.40
    if c2_passed: score += 0.20
    if c4_passed: score += 0.25
    if c3_passed: score += 0.05
    score += 0.10 * efficiency
    score -= 0.05 * wrong_fixes
    score = round(max(0.0, min(1.0, score)), 4)

    checks = [
        GraderCheck(name="all_services_healthy",  passed=c1_passed, weight=0.40, detail="All 4 services must be healthy" if c1_passed else "Some services still degraded."),
        GraderCheck(name="investigated_root_cause",passed=c2_passed, weight=0.20, detail="Agent investigated the root cause service before fixing." if c2_passed else "Agent did not investigate root cause service."),
        GraderCheck(name="correct_fix_sequence",   passed=c4_passed, weight=0.25, detail=f"All {total_required} correct fixes applied." if c4_passed else f"Only {correct_fixes}/{total_required} correct fixes applied."),
        GraderCheck(name="no_wrong_fixes",         passed=c3_passed, weight=0.05, detail="No trap actions taken." if c3_passed else f"{wrong_fixes} wrong fix attempt(s) penalized."),
        GraderCheck(name="efficiency",             passed=efficiency > 0.5, weight=0.10, detail=f"Resolved in {ticks}/{max_ticks} ticks (efficiency={efficiency:.2f})."),
    ]
    passed = all_healthy and correct_fixes == total_required
    return GraderReport(
        scenario_id=scenario_id, passed=passed, score=score,
        message="Incident fully resolved." if passed else "Incident not fully resolved.",
        checks=checks
    )
```

---

## [IMPLEMENT] File 4: sre_env/server/grader.py

```python
# sre_env/server/grader.py
"""Per-step reward computation (dense reward shaping)."""

from __future__ import annotations


class SREGrader:
    """
    Dense reward function. Called every step.

    Design principles (from RL reward shaping research):
    - Potential-based shaping: rewards reflect PROGRESS toward goal, not just goal
    - Investigation bonus: agent learns to diagnose before acting
    - Trap penalty: wrong fixes cause negative reward, not just wasted ticks
    - Tick penalty: time pressure forces efficiency
    - Terminal bonus: large reward for full resolution
    - Normalize: all values bounded to prevent reward hacking

    Per-step reward breakdown:
      -0.05    always (tick penalty — time is money)
      +0.15    investigating the root cause service (once per episode)
      +0.08    investigating any OTHER service (information value)
      +0.35    applying a correct fix in the right sequence
      +0.20    partial alert clearance (per alert cleared)
      -0.20    trap action (wrong fix that is explicitly harmful)
      +1.00    terminal bonus — all services healthy
    """

    def compute(
        self,
        action_tool: str,
        action_service: str,
        root_cause_service: str,
        investigated_root_cause_before: bool,
        was_correct_fix: bool,
        was_trap_action: bool,
        alerts_cleared: int,
        all_healthy: bool,
    ) -> float:
        reward = 0.0

        # Always: tick penalty
        reward -= 0.05

        # Investigation rewards
        if action_tool in ("get_logs", "get_metrics", "get_dependencies"):
            if action_service == root_cause_service and not investigated_root_cause_before:
                reward += 0.15  # First time looking at root cause — big bonus
            elif action_service != root_cause_service:
                reward += 0.08  # Looking at non-root-cause — smaller bonus (still useful)

        # Fix rewards/penalties
        if was_correct_fix:
            reward += 0.35
        elif was_trap_action:
            reward -= 0.20

        # Alert clearance reward
        reward += alerts_cleared * 0.20

        # Terminal bonus
        if all_healthy:
            reward += 1.00

        # Clamp to reasonable range
        return round(max(-1.0, min(2.0, reward)), 4)
```

---

## [IMPLEMENT] File 5: sre_env/server/environment.py

```python
# sre_env/server/environment.py
"""Core SRE state machine — the heart of the environment."""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import SREAction, SREObservation, SREState, ServiceInfo, Alert
    from .grader import SREGrader
    from .challenge import set_runtime_progress, grade_episode
except ImportError:
    from models import SREAction, SREObservation, SREState, ServiceInfo, Alert  # type: ignore
    from server.grader import SREGrader  # type: ignore
    from server.challenge import set_runtime_progress, grade_episode  # type: ignore

SCENARIOS_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"
MAX_TICKS = 20


def _load_scenarios() -> list:
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


SCENARIOS = {s["id"]: s for s in _load_scenarios()}


class SREEnvironment(Environment[SREAction, SREObservation, SREState]):
    """
    SRE Incident Response Environment.

    One episode = one incident scenario. Agent investigates and fixes
    a broken 4-service system within MAX_TICKS steps.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._grader = SREGrader()
        self._ep: dict = {}
        self._init_blank_episode()

    def _init_blank_episode(self) -> None:
        first = SCENARIOS["easy_001"]
        self._ep = self._make_episode(first)

    def _make_episode(self, scenario: dict, episode_id: str | None = None) -> dict:
        svcs = {
            name: ServiceInfo(name=name, **data)
            for name, data in scenario["initial_state"].items()
        }
        alerts = [Alert(**a) for a in scenario["initial_alerts"]]
        return {
            "episode_id":     episode_id or str(uuid.uuid4()),
            "scenario":       scenario,
            "difficulty":     scenario["difficulty"],
            "tick":           0,
            "max_ticks":      MAX_TICKS,
            "services":       svcs,
            "alerts":         alerts,
            "cumulative_reward": 0.0,
            "investigated_root_cause": False,
            "fix_attempts":   0,
            "wrong_fix_attempts": 0,
            "correct_fixes_applied": 0,
            "fix_sequence_progress": 0,
            "episode_complete": False,
            "success":        False,
            "failure_reason": None,
        }

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> SREObservation:
        difficulty = kwargs.get("difficulty", "easy")
        scenario_id = kwargs.get("scenario_id")

        if scenario_id:
            scenario = SCENARIOS[scenario_id]
        else:
            import random
            candidates = [s for s in SCENARIOS.values() if s["difficulty"] == difficulty]
            scenario = random.choice(candidates)

        self._ep = self._make_episode(scenario, episode_id)
        set_runtime_progress(self._state_dict())
        return self._build_obs(
            last_action_result="Incident detected. Investigate and restore all services.",
            tool_output=None, reward=0.0, done=False
        )

    def step(self, action: SREAction, timeout_s: float | None = None, **kwargs: Any) -> SREObservation:
        ep = self._ep
        if ep["episode_complete"]:
            return self._build_obs("Episode complete. Call reset().", None, 0.0, True)

        ep["tick"] += 1
        prev_alert_count = len([a for a in ep["alerts"] if a.active])
        last_result = ""
        tool_output = None
        was_correct_fix = False
        was_trap_action = False

        # ── Route action ──────────────────────────────────────────────────────
        if action.tool in ("get_logs", "get_metrics", "get_dependencies"):
            tool_output, last_result = self._handle_investigation(action)
        else:
            last_result, was_correct_fix, was_trap_action = self._handle_fix(action)

        # ── Alert clearance count ─────────────────────────────────────────────
        new_alert_count = len([a for a in ep["alerts"] if a.active])
        alerts_cleared = max(0, prev_alert_count - new_alert_count)

        # ── Compute reward ────────────────────────────────────────────────────
        reward = self._grader.compute(
            action_tool=action.tool,
            action_service=action.service,
            root_cause_service=ep["scenario"]["root_cause_service"],
            investigated_root_cause_before=ep["investigated_root_cause"],
            was_correct_fix=was_correct_fix,
            was_trap_action=was_trap_action,
            alerts_cleared=alerts_cleared,
            all_healthy=self._all_healthy(),
        )

        # Mark investigation flag AFTER reward so first-time bonus applies once
        if action.tool in ("get_logs", "get_metrics", "get_dependencies"):
            if action.service == ep["scenario"]["root_cause_service"]:
                ep["investigated_root_cause"] = True

        ep["cumulative_reward"] += reward

        # ── Terminal check ────────────────────────────────────────────────────
        done = False
        if self._all_healthy():
            ep["success"] = True
            ep["episode_complete"] = True
            done = True
            last_result += " ✅ ALL SERVICES HEALTHY — Incident resolved!"
        elif ep["tick"] >= ep["max_ticks"]:
            ep["episode_complete"] = True
            ep["failure_reason"] = f"Timeout: {ep['max_ticks']} ticks used."
            done = True
            last_result += " ⏰ TIME LIMIT REACHED."

        set_runtime_progress(self._state_dict())
        return self._build_obs(last_result.strip(), tool_output, reward, done)

    @property
    def state(self) -> SREState:
        return SREState(**self._state_dict())

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SREEnvironment",
            description="SRE Incident Response Commander: 4-service dependency graph with trap states, dense rewards, and 3 difficulty levels.",
            version="1.0.0",
            author="SRE-Env Team",
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _handle_investigation(self, action: SREAction) -> tuple[str, str]:
        ep = self._ep
        scenario = ep["scenario"]
        svc = action.service

        if action.tool == "get_logs":
            output = scenario["logs"].get(svc, f"No logs available for {svc}.")
            result = f"[get_logs] Logs retrieved for {svc}."

        elif action.tool == "get_metrics":
            if not action.metric:
                return "ERROR: metric parameter required for get_metrics.", "ERROR: metric required."
            metrics = scenario["metrics"].get(svc, {})
            output = metrics.get(action.metric, f"No metric '{action.metric}' for {svc}.")
            result = f"[get_metrics] {action.metric} for {svc} retrieved."

        elif action.tool == "get_dependencies":
            deps = scenario["dependencies"]
            output = deps.get(svc, f"No dependency info for {svc}.")
            result = f"[get_dependencies] Dependency info for {svc} retrieved."

        else:
            output, result = "Unknown tool.", "Unknown tool."

        return output, result

    def _handle_fix(self, action: SREAction) -> tuple[str, bool, bool]:
        ep = self._ep
        scenario = ep["scenario"]
        ep["fix_attempts"] += 1
        progress = ep["fix_sequence_progress"]
        correct_seq = scenario["correct_fix_sequence"]

        # Check correct sequence
        if progress < len(correct_seq):
            expected = correct_seq[progress]
            if (action.tool == expected["tool"] and
                action.service == expected["service"] and
                (expected.get("version") is None or action.version == expected.get("version"))):

                ep["fix_sequence_progress"] += 1
                ep["correct_fixes_applied"] += 1
                msg = self._apply_resolution(progress)
                return msg, True, False

        # Check trap actions
        for trap in scenario.get("trap_actions", []):
            ta = trap["action"]
            if (action.tool == ta["tool"] and action.service == ta["service"] and
                ta.get("version") == action.version):
                ep["wrong_fix_attempts"] += 1
                return trap["result"], False, True

        # Irrelevant action
        return (
            f"{action.tool} on {action.service}: No effect on current incident. "
            "System state unchanged.",
            False, False
        )

    def _apply_resolution(self, step_idx: int) -> str:
        ep = self._ep
        scenario = ep["scenario"]
        resolution = scenario.get("resolution", {})
        steps_done = ep["fix_sequence_progress"]
        total_steps = len(scenario["correct_fix_sequence"])

        for key, res in resolution.items():
            if steps_done == total_steps and "final_state" in res:
                for svc_name, vals in res["final_state"].items():
                    ep["services"][svc_name] = ServiceInfo(name=svc_name, **vals)
                ep["alerts"] = [a for a in ep["alerts"] if ep["services"][a.service].status != "healthy"]
                return res.get("message", "Fix applied. All services recovering.")
            elif "intermediate_state" in res and steps_done < total_steps:
                for svc_name, vals in res["intermediate_state"].items():
                    ep["services"][svc_name] = ServiceInfo(name=svc_name, **vals)
                ep["alerts"] = [a for a in ep["alerts"] if ep["services"][a.service].status != "healthy"]
                return res.get("message", "Partial fix applied. Continue.")

        return "Fix step applied."

    def _all_healthy(self) -> bool:
        return all(s.status == "healthy" for s in self._ep["services"].values())

    def _state_dict(self) -> dict:
        ep = self._ep
        return {
            "episode_id":      ep["episode_id"],
            "step_count":      ep["tick"],
            "difficulty":      ep["difficulty"],
            "scenario_id":     ep["scenario"]["id"],
            "cumulative_reward": ep["cumulative_reward"],
            "current_tick":    ep["tick"],
            "max_ticks":       ep["max_ticks"],
            "investigated_root_cause_service": ep["investigated_root_cause"],
            "fix_attempts":    ep["fix_attempts"],
            "wrong_fix_attempts": ep["wrong_fix_attempts"],
            "correct_fixes_applied": ep["correct_fixes_applied"],
            "all_services_healthy": self._all_healthy(),
            "episode_complete": ep["episode_complete"],
        }

    def _build_obs(self, last_action_result: str, tool_output, reward: float, done: bool) -> SREObservation:
        ep = self._ep
        active = [a for a in ep["alerts"] if a.active]
        return SREObservation(
            tick=ep["tick"],
            max_ticks=ep["max_ticks"],
            difficulty=ep["difficulty"],
            services=ep["services"],
            active_alerts=active,
            last_action_result=last_action_result,
            tool_output=tool_output,
            episode_complete=ep["episode_complete"],
            success=ep.get("success", False),
            failure_reason=ep.get("failure_reason"),
            reward=reward,
            done=done,
        )
```

---

## [IMPLEMENT] File 6: sre_env/server/app.py

```python
# sre_env/server/app.py
"""FastAPI app with standard OpenEnv routes + extra REST endpoints for debuggability."""

from __future__ import annotations
import os
import argparse
from fastapi import HTTPException

try:
    from openenv.core.env_server.http_server import create_app
    from ..models import (
        ScenarioCatalog, GraderReport, BaselineCatalog, RuntimeStatus,
        SREAction, SREObservation
    )
    from .challenge import (
        list_scenarios, list_baselines, current_runtime_progress, grade_episode
    )
    from .environment import SREEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app  # type: ignore
    from models import (  # type: ignore
        ScenarioCatalog, GraderReport, BaselineCatalog, RuntimeStatus,
        SREAction, SREObservation
    )
    from server.challenge import (  # type: ignore
        list_scenarios, list_baselines, current_runtime_progress, grade_episode
    )
    from server.environment import SREEnvironment  # type: ignore


app = create_app(
    SREEnvironment,
    SREAction,
    SREObservation,
    env_name="sre_env",
    max_concurrent_envs=1,
)


@app.get("/tasks", response_model=ScenarioCatalog, tags=["challenge"])
def tasks(difficulty: str | None = None) -> ScenarioCatalog:
    """List all incident scenarios, optionally filtered by difficulty."""
    try:
        return list_scenarios(difficulty=difficulty)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/grader", response_model=GraderReport, tags=["challenge"])
def grader(scenario_id: str | None = None) -> GraderReport:
    """Return current grader score for active episode."""
    progress = current_runtime_progress()
    if scenario_id:
        progress["scenario_id"] = scenario_id
    try:
        return grade_episode(progress)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/baseline", response_model=BaselineCatalog, tags=["challenge"])
def baseline(scenario_id: str | None = None) -> BaselineCatalog:
    """Return optimal baseline trajectories (what a perfect agent does)."""
    try:
        return list_baselines(scenario_id=scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/status", response_model=RuntimeStatus, tags=["challenge"])
def status() -> RuntimeStatus:
    """Return live runtime state and grader outcome for the active episode."""
    from ..models import SREState
    progress = current_runtime_progress()
    report = grade_episode(progress)
    try:
        state = SREState(**progress)
    except Exception:
        from models import SREState  # type: ignore
        state = SREState(**progress)
    return RuntimeStatus(progress=state, grader=report)


@app.get("/health")
def health():
    return {"status": "ok", "environment": "sre_env", "version": "1.0.0"}


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
```

---

## [IMPLEMENT] File 7: sre_env/client.py

```python
# sre_env/client.py
"""Typed WebSocket client for the SRE Incident Response environment."""

from __future__ import annotations
from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import SREAction, SREObservation, SREState, ServiceInfo, Alert


class SREEnv(EnvClient[SREAction, SREObservation, SREState]):
    """
    WebSocket client for the SRE environment.

    Async usage:
        async with SREEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset(difficulty="medium")
            result = await env.step(SREAction(tool="get_logs", service="database"))

    Sync usage:
        with SREEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset(difficulty="easy")
            result = env.step(SREAction(tool="restart", service="database"))
    """

    DEFAULT_BASE_URL = "http://localhost:8000"
    HF_SPACE_URL = "https://YOUR_HF_USERNAME-sre-env.hf.space"  # ← UPDATE THIS

    def _step_payload(self, action: SREAction) -> Dict[str, Any]:
        return {
            "tool": action.tool,
            "service": action.service,
            "metric": action.metric,
            "replicas": action.replicas,
            "version": action.version,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SREObservation]:
        obs_data = payload.get("observation", {})
        services_raw = obs_data.get("services", {})
        services = {k: ServiceInfo(**v) if isinstance(v, dict) else v for k, v in services_raw.items()}
        alerts = [Alert(**a) if isinstance(a, dict) else a for a in obs_data.get("active_alerts", [])]
        observation = SREObservation(
            tick=obs_data.get("tick", 0),
            max_ticks=obs_data.get("max_ticks", 20),
            difficulty=obs_data.get("difficulty", "easy"),
            services=services,
            active_alerts=alerts,
            last_action_result=obs_data.get("last_action_result", ""),
            tool_output=obs_data.get("tool_output"),
            episode_complete=payload.get("done", False),
            success=obs_data.get("success", False),
            failure_reason=obs_data.get("failure_reason"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SREState:
        return SREState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "easy"),
            scenario_id=payload.get("scenario_id", "easy_001"),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            current_tick=payload.get("current_tick", 0),
            max_ticks=payload.get("max_ticks", 20),
            investigated_root_cause_service=payload.get("investigated_root_cause_service", False),
            fix_attempts=payload.get("fix_attempts", 0),
            wrong_fix_attempts=payload.get("wrong_fix_attempts", 0),
            correct_fixes_applied=payload.get("correct_fixes_applied", 0),
            all_services_healthy=payload.get("all_services_healthy", False),
            episode_complete=payload.get("episode_complete", False),
        )
```

---

## [IMPLEMENT] File 8: sre_env/scripts/baseline_agent.py

```python
#!/usr/bin/env python3
# sre_env/scripts/baseline_agent.py
"""
Baseline agent using Llama-3.1-8B-Instant via Groq API.

Run:
    python -m sre_env.scripts.baseline_agent

Expected: Easy >0.7, Medium >0.5, Hard >0.4
"""

from __future__ import annotations
import os, json, asyncio
from openai import OpenAI
from sre_env.client import SREEnv
from sre_env.models import SREAction

# ── API configuration ──────────────────────────────────────────────────────
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.1-8b-instant"

groq = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer responding to a LIVE production incident.

SERVICES: api-gateway, cache, database, worker
DEPENDENCY ORDER: internet → api-gateway → cache → database → worker

AVAILABLE TOOLS (respond with JSON only):
  {"tool": "get_logs",        "service": "<name>"}
  {"tool": "get_metrics",     "service": "<name>", "metric": "<cpu|memory|latency|error_rate|throughput>"}
  {"tool": "get_dependencies","service": "<name>"}
  {"tool": "restart",         "service": "<name>"}
  {"tool": "scale",           "service": "<name>", "replicas": <1-5>}
  {"tool": "rollback",        "service": "<name>", "version": "previous"}

STRATEGY:
1. Look at active_alerts first — identify which services are affected
2. INVESTIGATE before fixing — use get_logs or get_metrics on the most suspicious service
3. Think about the dependency graph — the visibly broken service may not be the root cause
4. Fix ROOT CAUSE first. Then fix downstream victims.
5. A wrong fix may make things WORSE.

CRITICAL: Respond with ONLY valid JSON. No explanation. No markdown. Just the JSON object.
"""

def _build_user_message(obs: dict) -> str:
    alerts = obs.get("active_alerts", [])
    services = obs.get("services", {})
    tick = obs.get("tick", 0)
    max_ticks = obs.get("max_ticks", 20)
    last = obs.get("last_action_result", "")
    tool_out = obs.get("tool_output")

    msg = f"=== TICK {tick}/{max_ticks} ===\n\n"
    msg += "ACTIVE ALERTS:\n"
    for a in alerts:
        msg += f"  [{a['severity'].upper()}] {a['service']}: {a['message']}\n"
    if not alerts:
        msg += "  (none — all clear)\n"

    msg += "\nSERVICE STATUS:\n"
    for name, svc in services.items():
        s = svc if isinstance(svc, dict) else svc.model_dump()
        msg += f"  {name:14s} {s['status']:12s} cpu={s['cpu_pct']}% mem={s['memory_pct']}% err={s['error_rate_pct']}%\n"

    if last:
        msg += f"\nLAST ACTION RESULT:\n{last}\n"
    if tool_out:
        msg += f"\nTOOL OUTPUT:\n{tool_out}\n"

    msg += "\nWhat is your next action? JSON only:"
    return msg

def _call_agent(history: list) -> dict:
    resp = groq.chat.completions.create(
        model=MODEL, messages=history, temperature=0.1, max_tokens=150
    )
    raw = resp.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())

async def run_episode(difficulty: str, base_url: str = "http://localhost:8000") -> float:
    print(f"\n{'='*60}\n  Episode: {difficulty.upper()}\n{'='*60}")
    total_reward = 0.0

    async with SREEnv(base_url=base_url) as env:
        obs = await env.reset(difficulty=difficulty)
        obs_dict = obs.model_dump()
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False

        while not done:
            user_msg = _build_user_message(obs_dict)
            history.append({"role": "user", "content": user_msg})

            try:
                action_dict = _call_agent(history)
                print(f"  Tick {obs_dict['tick']:2d} → {json.dumps(action_dict)}")
            except Exception as e:
                print(f"  Parse error: {e} — defaulting to get_logs database")
                action_dict = {"tool": "get_logs", "service": "database"}

            action = SREAction(**action_dict)
            result = await env.step(action)
            obs, reward, done = result.observation, result.reward, result.done
            obs_dict = obs.model_dump()
            total_reward += reward
            history.append({"role": "assistant", "content": json.dumps(action_dict)})

            if done:
                status = "✅ SUCCESS" if obs_dict.get("success") else "❌ FAILED"
                print(f"\n  {status} | total_reward={total_reward:.3f}")
                if obs_dict.get("failure_reason"):
                    print(f"  Reason: {obs_dict['failure_reason']}")

    return total_reward

async def main():
    results = {}
    for d in ["easy", "medium", "hard"]:
        results[d] = await run_episode(d)

    print(f"\n{'='*60}\n  BASELINE RESULTS SUMMARY\n{'='*60}")
    for d, r in results.items():
        bar = "█" * max(0, int((r + 1) * 8))
        print(f"  {d.upper():6s}: reward={r:+.3f}  {bar}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## [IMPLEMENT] File 9: sre_env/scripts/learning_curve.py

```python
#!/usr/bin/env python3
# sre_env/scripts/learning_curve.py
"""
Learning curve comparison: BASELINE (no memory) vs FEW-SHOT (learns from history).

This is the visual proof that the environment produces a genuine RL training signal.
Saves learning_curve.png in the project root.

Run:
    python -m sre_env.scripts.learning_curve
"""

from __future__ import annotations
import os, json, asyncio, random
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from openai import OpenAI
from sre_env.client import SREEnv
from sre_env.models import SREAction

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
groq = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL = "llama-3.1-8b-instant"
N_EPISODES = 20  # episodes per difficulty per agent type
BASE_URL = "http://localhost:8000"

SYSTEM_BASELINE = """You are an SRE. Respond with JSON tool calls only.
Tools: get_logs, get_metrics, get_dependencies, restart, scale, rollback
Services: api-gateway, cache, database, worker
JSON format: {"tool": "...", "service": "..."}
No memory of past episodes."""

def system_fewshot(history_text: str) -> str:
    return f"""You are an SRE improving from past experience.
Tools: get_logs, get_metrics, get_dependencies, restart, scale, rollback
Services: api-gateway, cache, database, worker
JSON format: {{"tool": "...", "service": "..."}}

YOUR PAST EPISODE OUTCOMES:
{history_text}

Use patterns from past successes. Avoid past mistakes. Respond JSON only."""

def obs_to_text(obs: dict) -> str:
    alerts = obs.get("active_alerts", [])
    svcs = obs.get("services", {})
    lines = [f"Tick {obs.get('tick',0)}/{obs.get('max_ticks',20)}"]
    for a in alerts:
        lines.append(f"ALERT [{a['severity']}] {a['service']}: {a['message']}")
    for name, svc in svcs.items():
        s = svc if isinstance(svc, dict) else svc.model_dump()
        lines.append(f"{name}: {s['status']} cpu={s['cpu_pct']}% err={s['error_rate_pct']}%")
    if obs.get("tool_output"):
        lines.append(f"TOOL: {obs['tool_output'][:200]}")
    if obs.get("last_action_result"):
        lines.append(f"RESULT: {obs['last_action_result'][:150]}")
    return "\n".join(lines)

def call_groq(messages: list) -> dict:
    try:
        resp = groq.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.2, max_tokens=120
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {"tool": "get_logs", "service": random.choice(["database","cache","worker","api-gateway"])}

async def run_episode_simple(difficulty: str, system_msg: str) -> tuple[float, str]:
    """Run one episode, return (total_reward, episode_summary_text)."""
    async with SREEnv(base_url=BASE_URL) as env:
        obs = await env.reset(difficulty=difficulty)
        obs_dict = obs.model_dump()
        messages = [{"role": "system", "content": system_msg}]
        total_reward = 0.0
        actions_taken = []
        done = False

        while not done:
            user_msg = obs_to_text(obs_dict) + "\nNext action JSON:"
            messages.append({"role": "user", "content": user_msg})
            action_dict = call_groq(messages)
            actions_taken.append(action_dict)
            messages.append({"role": "assistant", "content": json.dumps(action_dict)})

            try:
                action = SREAction(**action_dict)
                result = await env.step(action)
                obs, reward, done = result.observation, result.reward, result.done
                obs_dict = obs.model_dump()
                total_reward += reward
            except Exception:
                break

        success = obs_dict.get("success", False)
        summary = (
            f"Episode: difficulty={difficulty} reward={total_reward:.2f} "
            f"success={success} actions={len(actions_taken)} "
            f"last_actions={[a.get('tool')+'('+a.get('service','')+')' for a in actions_taken[-3:]]}"
        )
        return total_reward, summary

async def run_comparison(difficulty: str):
    print(f"\n  Running {N_EPISODES} episodes × 2 agents for difficulty={difficulty}...")

    baseline_rewards = []
    fewshot_rewards = []
    fewshot_history: List[str] = []

    for ep in range(N_EPISODES):
        # Baseline agent: no memory
        r_base, _ = await run_episode_simple(difficulty, SYSTEM_BASELINE)
        baseline_rewards.append(r_base)

        # Few-shot agent: fed last 3 episode summaries
        history_text = "\n".join(fewshot_history[-3:]) if fewshot_history else "No history yet."
        r_few, summary = await run_episode_simple(difficulty, system_fewshot(history_text))
        fewshot_rewards.append(r_few)
        fewshot_history.append(summary)

        print(f"    ep {ep+1:2d}/{N_EPISODES}  baseline={r_base:+.2f}  fewshot={r_few:+.2f}")
        await asyncio.sleep(0.3)  # rate limit courtesy

    return baseline_rewards, fewshot_rewards

def smooth(values: list, window: int = 3) -> list:
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        out.append(np.mean(values[lo:hi]))
    return out

async def main():
    difficulties = ["easy", "medium", "hard"]
    all_results = {}
    for d in difficulties:
        all_results[d] = await run_comparison(d)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")

    colors = {
        "baseline": "#ff6b6b",
        "fewshot":  "#4ecdc4",
    }
    titles = {"easy": "Easy — DB Crash", "medium": "Medium — Cache Cascade", "hard": "Hard — Bad Deploy"}

    for ax, difficulty in zip(axes, difficulties):
        ax.set_facecolor("#161b22")
        baseline_r, fewshot_r = all_results[difficulty]
        x = list(range(1, N_EPISODES + 1))

        # Raw (faded)
        ax.plot(x, baseline_r, color=colors["baseline"], alpha=0.25, linewidth=1)
        ax.plot(x, fewshot_r,  color=colors["fewshot"],  alpha=0.25, linewidth=1)

        # Smoothed (bold)
        ax.plot(x, smooth(baseline_r), color=colors["baseline"], linewidth=2.5,
                label=f"Baseline (avg={np.mean(baseline_r):.2f})")
        ax.plot(x, smooth(fewshot_r),  color=colors["fewshot"],  linewidth=2.5,
                label=f"Few-Shot  (avg={np.mean(fewshot_r):.2f})")

        # Improvement arrow
        improvement = np.mean(fewshot_r[-5:]) - np.mean(baseline_r[:5])
        if improvement > 0:
            ax.annotate(f"+{improvement:.2f}", xy=(N_EPISODES * 0.85, max(fewshot_r[-3:])),
                       fontsize=10, color="#4ecdc4", fontweight="bold")

        ax.axhline(0, color="#444", linestyle="--", linewidth=0.8)
        ax.set_title(titles[difficulty], color="white", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Episode", color="#aaa", fontsize=10)
        ax.set_ylabel("Total Reward", color="#aaa", fontsize=10)
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.legend(facecolor="#1c2128", labelcolor="white", fontsize=9, framealpha=0.8)
        ax.grid(alpha=0.15, color="#444")

    fig.suptitle(
        "SRE-Env: Baseline vs Few-Shot Agent — Learning Curve Comparison",
        color="white", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out_path = "learning_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  ✅ Saved: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## [IMPLEMENT] File 10: sre_env/tests/test_environment.py

```python
# sre_env/tests/test_environment.py
import pytest
from sre_env.server.environment import SREEnvironment
from sre_env.models import SREAction


@pytest.fixture
def env():
    e = SREEnvironment()
    return e


def test_reset_returns_observation(env):
    obs = env.reset(difficulty="easy", scenario_id="easy_001")
    assert obs.tick == 0
    assert len(obs.active_alerts) > 0
    assert obs.difficulty == "easy"
    assert not obs.episode_complete


def test_step_increments_tick(env):
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="get_logs", service="database"))
    assert obs.tick == 1


def test_investigation_returns_tool_output(env):
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="get_logs", service="database"))
    assert obs.tool_output is not None
    assert "OutOfMemoryError" in obs.tool_output


def test_correct_fix_resolves_easy(env):
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="restart", service="database"))
    assert obs.episode_complete is True
    assert obs.success is True


def test_trap_action_negative_reward(env):
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="restart", service="api-gateway"))
    assert obs.reward < 0
    assert obs.episode_complete is False


def test_investigating_root_cause_positive_reward(env):
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="get_logs", service="database"))
    assert obs.reward > 0  # investigation bonus (+0.15) > tick penalty (-0.05)


def test_medium_trap_fails_again(env):
    env.reset(difficulty="medium", scenario_id="medium_001")
    obs = env.step(SREAction(tool="restart", service="database"))
    assert obs.episode_complete is False
    assert "cache" in obs.last_action_result.lower() or "root" in obs.last_action_result.lower()


def test_medium_correct_sequence(env):
    env.reset(difficulty="medium", scenario_id="medium_001")
    env.step(SREAction(tool="restart", service="cache"))
    obs = env.step(SREAction(tool="restart", service="database"))
    assert obs.success is True
    assert obs.episode_complete is True


def test_hard_rollback_first(env):
    env.reset(difficulty="hard", scenario_id="hard_001")
    env.step(SREAction(tool="rollback", service="worker", version="previous"))
    env.step(SREAction(tool="restart",  service="database"))
    obs = env.step(SREAction(tool="restart",  service="api-gateway"))
    assert obs.success is True


def test_state_metadata(env):
    env.reset(difficulty="hard", scenario_id="hard_001")
    state = env.state
    assert state.difficulty == "hard"
    assert state.step_count == 0


def test_timeout(env):
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = None
    for _ in range(21):
        obs = env.step(SREAction(tool="get_metrics", service="cache", metric="cpu"))
        if obs.episode_complete:
            break
    assert obs.episode_complete is True


def test_grader_score_perfect_easy(env):
    from sre_env.server.challenge import grade_episode
    env.reset(difficulty="easy", scenario_id="easy_001")
    env.step(SREAction(tool="get_logs", service="database"))
    env.step(SREAction(tool="restart",  service="database"))
    score_data = grade_episode(env._state_dict())
    assert score_data.score >= 0.7
```

---

## [IMPLEMENT] File 11: sre_env/tests/test_http_routes.py

```python
# sre_env/tests/test_http_routes.py
from fastapi.testclient import TestClient
from sre_env.server.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["environment"] == "sre_env"


def test_tasks_all():
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    assert data["environment"] == "sre_env"
    assert len(data["scenarios"]) == 3
    assert sorted(data["available_difficulties"]) == ["easy", "hard", "medium"]


def test_tasks_filtered_easy():
    r = client.get("/tasks", params={"difficulty": "easy"})
    assert r.status_code == 200
    data = r.json()
    assert all(s["difficulty"] == "easy" for s in data["scenarios"])


def test_tasks_invalid_difficulty():
    r = client.get("/tasks", params={"difficulty": "impossible"})
    assert r.status_code == 404


def test_baseline_all():
    r = client.get("/baseline")
    assert r.status_code == 200
    assert len(r.json()["baselines"]) == 3


def test_baseline_filtered():
    r = client.get("/baseline", params={"scenario_id": "easy_001"})
    assert r.status_code == 200
    assert r.json()["baselines"][0]["scenario_id"] == "easy_001"


def test_grader_returns_report():
    r = client.get("/grader")
    assert r.status_code == 200
    data = r.json()
    assert "score" in data
    assert "checks" in data


def test_status():
    r = client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert data["environment"] == "sre_env"
    assert "progress" in data
    assert "grader" in data
```

---

## [IMPLEMENT] File 12: server/Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/sre_env

ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

CMD ["uvicorn", "sre_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## [IMPLEMENT] File 13: server/requirements.txt

```
openenv-core>=0.2.1
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
websockets>=12.0
python-multipart>=0.0.6
```

---

## [IMPLEMENT] File 14: pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sre-env"
version = "1.0.0"
description = "SRE Incident Response Commander — OpenEnv environment for RL agent training"
requires-python = ">=3.10"
dependencies = [
    "openenv-core>=0.2.1",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0.0",
    "websockets>=12.0",
    "openai>=1.0.0",
    "matplotlib>=3.7.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-asyncio>=0.21", "httpx>=0.24"]

[tool.hatch.build.targets.wheel]
packages = ["sre_env"]
```

---

## [IMPLEMENT] File 15: openenv.yaml

```yaml
name: sre-env
version: 1.0.0
description: >
  SRE Incident Response Commander: Train AI agents to diagnose and fix
  production incidents across a 4-service microservice architecture.
  Features dependency-graph trap states, dense rewards, 3 difficulty levels.
author: YOUR_NAME
license: MIT

environment:
  action_type: SREAction
  observation_type: SREObservation
  state_type: SREState
  max_steps: 20
  difficulties: [easy, medium, hard]
  reward_type: dense

huggingface:
  space_id: YOUR_HF_USERNAME/sre-env
  sdk: docker
  hardware: cpu-basic
```

---

## [IMPLEMENT] File 16: Makefile

```makefile
.PHONY: install dev test docker-build docker-run validate baseline learning-curve

install:
	pip install -e ".[dev]"

dev:
	ENABLE_WEB_INTERFACE=true uvicorn sre_env.server.app:app --reload --port 8000

test:
	pytest sre_env/tests/ -v --tb=short

docker-build:
	docker buildx build --platform linux/amd64 -t sre-env:latest .

docker-run:
	docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true sre-env:latest

validate:
	openenv validate .

baseline:
	python -m sre_env.scripts.baseline_agent

learning-curve:
	python -m sre_env.scripts.learning_curve
	open learning_curve.png
```

---

## [IMPLEMENT] File 17: README.md

```markdown
# SRE-Env: Incident Response Commander

An OpenEnv reinforcement learning environment where an AI agent plays an on-call SRE
engineer diagnosing and fixing a broken 4-service production system.

## What makes this hard (and interesting for RL)

- **4-service dependency graph**: `api-gateway → cache → database → worker`
- **Trap states**: the visibly broken service is often NOT the root cause
- **Multi-step fixes**: Medium requires 2 steps in order, Hard requires 3 steps in order
- **Dense rewards**: investigation is rewarded separately from fixing — the RL signal is rich

## Quick Start

```bash
pip install -e ".[dev]"
make dev          # server at http://localhost:8000/web
make test         # run all tests
make baseline     # Groq/Llama-3 agent demo
make learning-curve  # matplotlib comparison chart
```

## Scenarios

| ID | Difficulty | Root Cause | Trap |
|---|---|---|---|
| easy_001 | Easy | database OOM crash | restart api-gateway (does nothing) |
| medium_001 | Medium | cache OOM → database overload | restart database first (crashes again) |
| hard_001 | Hard | worker bad deploy → DB corruption | restart database first (worker re-corrupts it) |

## Reward Function

| Event | Reward |
|---|---|
| Each tick (time penalty) | -0.05 |
| First investigation of root cause service | +0.15 |
| Investigating any service | +0.08 |
| Correct fix in sequence | +0.35 |
| Trap action (wrong fix) | -0.20 |
| Alert cleared | +0.20 |
| All services healthy | +1.00 |

## API Endpoints

Beyond the standard OpenEnv WebSocket API:

- `GET /tasks` — list all scenarios (filter by `?difficulty=easy`)
- `GET /grader` — current grader score
- `GET /baseline` — optimal agent trajectories
- `GET /status` — live episode progress
- `GET /health` — server health check
```

---

## [CONTEXT] Build Order & Success Criteria

Build files in exactly this order:
1. `models.py` → 2. `data/scenarios.json` → 3. `server/challenge.py` → 4. `server/grader.py`
→ 5. `server/environment.py` → 6. `server/app.py` → 7. `client.py`
→ 8. `scripts/baseline_agent.py` → 9. `scripts/learning_curve.py`
→ 10. Tests → 11. Dockerfile, requirements, pyproject.toml, openenv.yaml, Makefile, README

**✅ Done when:**
1. `make dev` starts server, no errors, `http://localhost:8000/web` works
2. `make test` — all 12 tests pass
3. `make baseline` — Easy >0.7, Medium >0.5, Hard >0.4
4. `make learning-curve` — `learning_curve.png` saved showing fewshot line above baseline
5. `make docker-build` succeeds (linux/amd64 for HF Spaces compatibility)
6. `openenv validate .` passes

**❌ Do NOT build in V1:**
- CLI/SSH simulation
- Noisy 50-line logs with red herrings
- LLM judge / post-mortem grading
- Multi-agent
- Authentication middleware
- Persistent database storage

*End of prompt. Start with models.py.*
```
