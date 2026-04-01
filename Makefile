.PHONY: install dev test docker-build docker-run validate baseline learning-curve

install:
	python3 -m pip install -e ".[dev]"

dev:
	ENABLE_WEB_INTERFACE=true uvicorn server.app:app --reload --port 8000

test:
	pytest sre_env/tests -v --tb=short

docker-build:
	docker buildx build --platform linux/amd64 -f server/Dockerfile -t sre-env:latest .

docker-run:
	docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true sre-env:latest

validate:
	openenv validate .

baseline:
	python -m sre_env.scripts.baseline_agent

learning-curve:
	python -m sre_env.scripts.learning_curve
