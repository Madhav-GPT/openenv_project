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

This project now prefers your local Ollama model by default:

```bash
ollama serve
make baseline
make learning-curve
```

Defaults:

- Provider preference: local Ollama -> Groq -> heuristic fallback
- Local model: `qwen2.5:7b`
- Override model: `OLLAMA_MODEL=<other-model> make baseline`
- Override provider explicitly:
  `python -m sre_env.scripts.baseline_agent --provider ollama --model qwen2.5:7b`
- Quick local-model smoke for the learning-curve path:
  `python -m sre_env.scripts.learning_curve --provider ollama --model qwen2.5:7b --episodes 1`
- Optional saved image artifact:
  `python -m sre_env.scripts.learning_curve --provider ollama --model qwen2.5:7b --episodes 1 --save-plot`
- Full learning-curve runs with a local 7B model can take a while because they issue many sequential inference calls.

While the scripts run, Terminal 2 now shows a live dashboard with:

- side-by-side difficulty panels
- reward progression
- per-step and cumulative reward graphs
- model latency and token/sec
- current action, alerts, and service health
- baseline vs few-shot comparison during learning-curve runs

The live dashboard is now the default output. `learning_curve.png` is only written if you pass `--save-plot`.

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
