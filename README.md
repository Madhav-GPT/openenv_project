# SRE-Env: Incident Response Commander

An OpenEnv reinforcement-learning environment where an agent plays an on-call SRE
diagnosing and fixing a broken 4-service production system.

## What makes this hard

- 4-service dependency graph: `api-gateway -> cache -> database -> worker`
- Trap states: the visibly broken service is often not the root cause
- Multi-step fixes: medium requires 2 ordered fixes, hard requires 3
- Dense rewards: investigation is rewarded separately from remediation

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make dev
make test
```

The dev server exposes the standard OpenEnv API plus:

- `GET /tasks`
- `GET /grader`
- `GET /baseline`
- `GET /status`
- `GET /health`

## Baseline And Learning Curve

```bash
make baseline
make learning-curve
```

If `GROQ_API_KEY` is set, the baseline script can query Groq's OpenAI-compatible
chat model. Without that key, both scripts fall back to deterministic local policies
so the project still runs end to end.

## Scenarios

| ID | Difficulty | Root Cause | Trap |
| --- | --- | --- | --- |
| `easy_001` | Easy | database OOM crash | restarting `api-gateway` does nothing |
| `medium_001` | Medium | cache OOM -> database overload | restarting `database` first overloads it again |
| `hard_001` | Hard | worker bad deploy -> DB corruption | restarting `database` first lets worker re-corrupt it |

## Reward Function

| Event | Reward |
| --- | --- |
| Each tick | `-0.05` |
| First investigation of root cause | `+0.15` |
| Investigation of another service | `+0.08` |
| Correct fix in sequence | `+0.35` |
| Trap action | `-0.20` |
| Alert cleared | `+0.20` |
| All services healthy | `+1.00` |
# openenv_project
