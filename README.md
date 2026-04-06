# SRE-Env: Incident Response Commander

SRE-Env is an OpenEnv simulation where an AI agent handles a production incident across three connected phases:

1. **Infrastructure triage** — investigate logs, metrics, and dependencies to diagnose the root cause
2. **Security remediation** — classify vulnerability, apply the correct patch, verify it works
3. **Post-mortem reasoning** — explain root cause, attack vector, fix sequence, and prevention

Each incident has a causal security root cause hidden behind misleading symptoms. The most visibly broken service is often not the root cause.

## Environment Shape

- Services: `api-gateway → cache → database → worker`
- Difficulties: `easy` (15 ticks), `medium` (20 ticks), `hard` (25 ticks)
- Tasks: `easy_001`, `medium_001`, `hard_001`
- Scores: normalized to `0.0..1.0`

### Action Space

| Phase | Tools |
|-------|-------|
| Phase 1: Investigation | `get_logs`, `get_metrics`, `get_dependencies` |
| Phase 1: Remediation | `restart`, `scale`, `rollback` |
| Phase 2: Security | `classify_vuln`, `apply_patch`, `verify_patch` |
| Phase 3: Reasoning | `post_mortem` |

### Observation Space

- Tick budget and current difficulty
- Service health (status, cpu, memory, error rate, latency)
- Active alerts with severity
- Tool output and last action result
- Phase status: `phase`, `phase2_unlocked`, `phase2_complete`, `security_sub_quest`, `websec_state`
- Post-mortem status: `postmortem_available`, `postmortem_submitted`
- Scoring: `final_score`, `phase_scores`

### Scoring Breakdown

| Component | Max | How |
|-----------|-----|-----|
| Infrastructure | 0.50 | All services healthy + root cause investigated + fix sequence + efficiency |
| Security | 0.40 | Phase 2 unlocked + classify + correct patch + exploit blocked + functionality ok |
| Post-mortem | 0.30 | Deterministic keyword matching on root cause, attack vector, prevention, fix sequence |
| **Total** | **1.00** | Capped at 1.0 |

## Why This Is Hard

- The most visibly broken service is often **not** the root cause
- Medium and hard scenarios require **ordered fix sequences**
- Security is not bolted on — it **explains** the infrastructure failure
- The agent trades off exploration against per-tick penalties
- Trap actions punish the obvious-but-wrong fix (e.g., restarting the symptomatic service)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make dev          # Start the environment server
python inference.py   # Run the submission script
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | Yes | `http://127.0.0.1:11434/v1` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Yes | `qwen2.5:1.5b` | Model identifier |
| `HF_TOKEN` | Yes | `local` | HuggingFace / API key |
| `ENV_BASE_URL` | No | `http://127.0.0.1:8000` | Environment server URL |

## API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | Health check |
| `/metadata` | GET | Environment metadata |
| `/schema` | GET | Action/observation/state schemas |
| `/tasks` | GET | List scenarios |
| `/baseline` | GET | Baseline trajectories |
| `/grader` | GET | Grade current episode |
| `/status` | GET | Runtime status |
| `/unified-tasks` | GET | List unified scenarios |
| `/phase2-baseline` | GET | Phase 2 baselines |
| `/reset` | POST | Reset environment |
| `/step` | POST | Take action |
| `/state` | GET | Current state |

## Training (Optional)

Discover better action trajectories using few-shot learning:

```bash
make train-easy       # 50 iterations on easy scenario
make train-medium     # 75 iterations on medium scenario
make train-hard       # 100 iterations on hard scenario
```

The trainer uses in-context learning: it runs episodes, stores full trajectories, and injects the top-K best trajectories as few-shot examples in the system prompt. The `inference.py` script automatically loads the trained policy if available.

Output artifacts in `outputs/grpo_sre/`:
- `trained_policy.json` — best action sequences per scenario
- `trajectory_memory.json` — full trajectory memory for few-shot learning
- `reward_history.json` — training metrics

## Inference Contract

The root-level `inference.py` is the submission script. It:
- Uses the **OpenAI client** for all LLM calls
- Emits structured stdout: `[START]`, `[STEP]`, `[END]`
- Loads trained policies if available, falls back to optimal baselines
- Completes all 3 tasks within the 20-minute time limit

## Validation

```bash
make validate         # Local structure check (openenv validate .)
make pre-validate     # Full end-to-end validation
```

## Deployment

- Hugging Face Spaces: uses the root `Dockerfile`
- OpenEnv multi-mode: `server/app.py` wraps `sre_env/server/app.py`
- All scoring is deterministic — no external LLM needed for the environment
