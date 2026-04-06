# SRE-Env: Execution Guide

## Architecture (One Sentence)

**Environment** (FastAPI server) presents incidents → **Agent** (inference.py using OpenAI client) diagnoses and fixes them → **Environment** scores deterministically.

No external LLM needed for scoring. The environment is fully self-contained.

---

## Step-by-Step: New Terminal

```bash
# 1. Set up
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
python3 -m venv .venv
source .venv/bin/activate
make install

# 2. Start the environment server
make dev
# Server runs on http://127.0.0.1:8000

# 3. (Separate terminal) Start local LLM for testing
ollama serve
# Runs on http://127.0.0.1:11434

# 4. (Separate terminal) Run inference
source .venv/bin/activate
python inference.py
```

## What Each Make Target Does

| Command | What It Does | When To Use |
|---------|-------------|-------------|
| `make install` | `pip install -e ".[dev]"` | First time setup |
| `make dev` | Start FastAPI server on port 8000 | Always needed |
| `make validate` | `openenv validate .` — checks structure | Before submission |
| `make pre-validate` | Full end-to-end validation suite | Before submission |
| `python inference.py` | Runs the submission script | Testing/submission |
| `make train-easy` | Train on easy scenario (50 iterations) | Optional |
| `make train-medium` | Train on medium scenario (75 iterations) | Optional |
| `make train-hard` | Train on hard scenario (100 iterations) | Optional |

## Environment Variables

For **local testing** (with Ollama):
```bash
export API_BASE_URL=http://127.0.0.1:11434/v1
export MODEL_NAME=qwen2.5:1.5b
export HF_TOKEN=local
```

For **production** (competition judges):
```bash
export API_BASE_URL=<their-openai-endpoint>
export MODEL_NAME=<their-model>
export HF_TOKEN=<their-key>
```

The inference script uses `os.environ.get()` with sensible defaults, so you don't need to set anything for local testing if Ollama is running on the default port.

## How Scoring Works

The final score (0.0–1.0) is the sum of three components:

| Component | Max Score | What It Measures |
|-----------|-----------|-----------------|
| Infrastructure | 0.50 | All services healthy, root cause investigated, correct fix sequence, efficiency |
| Security | 0.40 | Vulnerability classified, correct patch applied, exploit blocked, functionality OK |
| Post-mortem | 0.30 | Root cause explanation, attack vector, prevention strategy, fix sequence quality |

**Total is capped at 1.0.** All scoring is deterministic — keyword matching and state checks, no LLM in the loop.

## How Training Works (Optional)

Training uses **in-context few-shot learning**:

1. Run episodes with the LLM against the environment
2. Store full (observation → action → reward) trajectories  
3. On subsequent episodes, inject the top-3 best trajectories as examples in the system prompt
4. The agent improves because it sees increasingly good examples

```bash
make train-easy    # Saves to outputs/grpo_sre/trained_policy.json
python inference.py  # Automatically loads trained policy if available
```

## Pre-Submission Checklist

```bash
# All of these must pass:
source .venv/bin/activate
make dev                    # Start server (in separate terminal)
python -m pytest sre_env/tests/ -v    # 40/40 tests
openenv validate .                     # Structure OK
openenv validate --url http://127.0.0.1:8000  # Runtime OK (6/6)
python inference.py                    # Produces [START]/[STEP]/[END]
python pre_submission_validate.py      # Full suite
```

## File Map

```
inference.py                 ← Submission script (OpenAI client, structured logs)
server/app.py                ← Top-level server entry (wraps sre_env)
Dockerfile                   ← HF Spaces deployment
openenv.yaml                 ← OpenEnv spec config
pyproject.toml               ← Package definition

sre_env/
├── models.py                ← Pydantic models (SREAction, SREObservation, SREState)
├── client.py                ← SREEnv client (step/reset via WebSocket)
├── data/
│   ├── scenarios.json       ← Phase 1 scenario definitions
│   └── unified_scenarios.json ← Full three-phase scenario definitions
├── server/
│   ├── app.py               ← FastAPI app with OpenEnv routes
│   ├── unified_environment.py ← Core environment (step/reset/state)
│   ├── judge.py             ← Deterministic post-mortem scorer
│   ├── grader.py            ← SRE reward shaping
│   ├── websec_grader.py     ← Security verification
│   └── challenge.py         ← Task catalog & grading endpoints
├── scripts/
│   ├── benchmark_policies.py ← Optimal/naive/random action sequences
│   ├── grpo_train.py        ← Few-shot trajectory learning trainer
│   ├── trajectory_memory.py ← Trajectory storage for few-shot learning
│   └── ...                  ← Demo scripts (baseline, dashboard, etc.)
└── tests/                   ← 40 tests
```
