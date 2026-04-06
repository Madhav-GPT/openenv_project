.PHONY: install dev test docker-build docker-run validate baseline \
	compare learning-curve train-easy train-medium train-hard train-all eval \
	judge-server dashboard demo demo-full clean pre-validate ping-space

install:
	python3 -m pip install -e ".[dev]"
	@echo "Dependencies installed"

dev:
	ENABLE_WEB_INTERFACE=true uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest sre_env/tests -v --tb=short

compare:
	python -m sre_env.scripts.three_agent_comparison

learning-curve:
	python -m sre_env.scripts.learning_curve --provider ollama --model qwen2.5:7b --episodes 1

train-easy:
	python -m sre_env.scripts.grpo_train --provider ollama --model qwen2.5:7b --difficulty easy --steps 50 --group-size 4 --output-dir outputs/grpo_sre

train-medium:
	python -m sre_env.scripts.grpo_train --provider ollama --model qwen2.5:7b --difficulty medium --steps 75 --group-size 4 --output-dir outputs/grpo_sre

train-hard:
	python -m sre_env.scripts.grpo_train --provider ollama --model qwen2.5:7b --difficulty hard --steps 100 --group-size 4 --output-dir outputs/grpo_sre

train-all: train-easy train-medium train-hard

eval:
	python -m sre_env.scripts.three_agent_comparison --use-trained-weights --n-episodes 8

judge-server:
	OLLAMA_HOST=0.0.0.0:11435 ollama serve

dashboard:
	python -m sre_env.scripts.learning_curve_dashboard --provider ollama --student-model qwen2.5:7b --episodes 5 --difficulties easy medium hard

baseline:
	python -m sre_env.scripts.baseline_agent --provider ollama --model qwen2.5:7b

docker-build:
	docker buildx build --platform linux/amd64 -t sre-env:latest .

docker-run:
	docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true sre-env:latest

validate:
	openenv validate .

pre-validate:
	python pre_submission_validate.py

ping-space:
	test -n "$(SPACE_URL)"
	curl -fsS "$(SPACE_URL)/health"

demo:
	python -m sre_env.scripts.demo_pipeline --quick

demo-full:
	python -m sre_env.scripts.demo_pipeline

clean:
	rm -rf outputs __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
