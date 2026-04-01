# Execution Guide

This runbook starts from a fresh shell in the project root:

`/Users/madhav_189/Documents/meta_hackathon/madhav_trial`

It assumes you want to use your local Ollama model `qwen2.5:7b`.

## 1. Open the project directory

```bash
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
```

## 2. Verify prerequisites

You need:

- Python 3.10+
- `ollama`
- `make`

Check them:

```bash
python3 --version
ollama --version
make --version
```

## 3. Start Ollama

Run this in Terminal 1:

```bash
ollama serve
```

Leave it running.

If it says the server is already running, that is fine.

## 4. Verify the model exists

Open Terminal 2 and run:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
ollama list
```

You should see `qwen2.5:7b`.

If you do not, pull it:

```bash
ollama pull qwen2.5:7b
```

## 5. Create and activate the virtual environment

```bash
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show the venv is active.

## 6. Install the project

```bash
make install
```

This installs the package plus test dependencies into `.venv`.

## 7. Run the automated tests

```bash
make test
```

Expected result:

- all tests pass

You can also run the validator:

```bash
openenv validate .
```

Expected result:

- `Ready for multi-mode deployment`

## 8. Start the environment server

Run this in Terminal 3:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
source .venv/bin/activate
make dev
```

Leave it running.

The server should start on:

- `http://127.0.0.1:8000`
- Web UI: `http://127.0.0.1:8000/web`

## 9. Smoke test the server

In Terminal 2:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
source .venv/bin/activate
curl http://127.0.0.1:8000/health
```

Expected JSON:

```json
{"status":"ok","environment":"sre_env","version":"1.0.0"}
```

You can also check the extra routes:

```bash
curl http://127.0.0.1:8000/tasks
curl http://127.0.0.1:8000/baseline
curl http://127.0.0.1:8000/status
```

## 10. Run the baseline agent with local Qwen

In Terminal 2:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
source .venv/bin/activate
python -m sre_env.scripts.baseline_agent --provider ollama --model qwen2.5:7b --base-url http://127.0.0.1:8000
```

Expected behavior:

- it opens a live terminal dashboard
- it runs easy, medium, and hard episodes
- it shows reward progression, token/sec, latency, alerts, and service state
- it finishes with a baseline summary

Notes:

- the project already prefers local Ollama by default
- the explicit flags above remove ambiguity

## 11. Run the learning-curve script

Quick smoke check first:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
source .venv/bin/activate
python -m sre_env.scripts.learning_curve --provider ollama --model qwen2.5:7b --episodes 1 --base-url http://127.0.0.1:8000
```

This is the recommended first run on a laptop.

It should produce:

- a live side-by-side dashboard in the terminal
- difficulty-by-difficulty reward graphs
- baseline vs few-shot progress panels
- model telemetry including token/sec and latency
- no image file by default; the terminal dashboard is the primary output

If you also want a saved image artifact:

```bash
python -m sre_env.scripts.learning_curve --provider ollama --model qwen2.5:7b --episodes 1 --base-url http://127.0.0.1:8000 --save-plot
```

That writes `learning_curve.png` in the project root.

Full run:

```bash
make learning-curve
```

Important:

- the full run can take a long time with a local 7B model
- it makes many sequential inference calls

## 12. Optional: open the generated artifact

```bash
ls -lh learning_curve.png
```

On macOS:

```bash
open learning_curve.png
```

Only run those if you used `--save-plot`.

## 13. Typical daily workflow

If the project is already installed, you usually only need:

Terminal 1:

```bash
ollama serve
```

Terminal 2:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
source .venv/bin/activate
make dev
```

Terminal 3:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/madhav_trial
source .venv/bin/activate
make test
python -m sre_env.scripts.baseline_agent --provider ollama --model qwen2.5:7b --base-url http://127.0.0.1:8000
```

## 14. Troubleshooting

### `ollama list` says server not responding

Start it:

```bash
ollama serve
```

### `qwen2.5:7b` is missing

Pull it:

```bash
ollama pull qwen2.5:7b
```

### `make install` fails

Make sure the venv is active:

```bash
source .venv/bin/activate
which python
which pip
```

They should point into `.venv`.

### `curl /health` fails

The server is not running. Start it again:

```bash
make dev
```

### Baseline or learning curve is very slow

This is normal with a local 7B model.

Use the shorter run:

```bash
python -m sre_env.scripts.learning_curve --provider ollama --model qwen2.5:7b --episodes 1 --base-url http://127.0.0.1:8000
```

### Model returns bad JSON occasionally

The scripts already fall back to a deterministic safe action when that happens.

### Port 8000 is busy

Run the server on another port:

```bash
uvicorn server.app:app --reload --host 127.0.0.1 --port 8001
```

Then point the scripts at that port:

```bash
python -m sre_env.scripts.baseline_agent --provider ollama --model qwen2.5:7b --base-url http://127.0.0.1:8001
python -m sre_env.scripts.learning_curve --provider ollama --model qwen2.5:7b --episodes 1 --base-url http://127.0.0.1:8001
```

## 15. Stop everything

In each running terminal, press:

```bash
Ctrl+C
```

That stops:

- `ollama serve`
- `make dev`
- any long-running baseline or learning-curve process
