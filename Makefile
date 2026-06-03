# Single entry point for running the app. Ports come from one source of truth:
# .env, falling back to .env.example (with a logged warning) when absent.
#
#   make up            # build + run the full stack in Docker on the .env ports
#   make down          # stop the Docker stack
#   make dev-backend   # run the FastAPI backend locally (no Docker)
#   make dev-frontend  # run the Vite dev server locally (no Docker)
#   make env-info      # print which env file is in use and its contents

SHELL := /bin/bash

# Resolve the configuration file once, at parse time.
ENV_FILE := $(shell [ -f .env ] && echo .env || echo .env.example)

# Warn (to stderr) when .env is missing and we fall back to .env.example.
define ENV_FALLBACK_WARN
	@if [ ! -f .env ]; then \
		echo "[config] .env non trovato — uso il fallback .env.example" >&2; \
	fi
endef

.PHONY: up down build dev-backend dev-frontend env-info

env-info:
	$(ENV_FALLBACK_WARN)
	@echo "Config: $(ENV_FILE)"
	@cat $(ENV_FILE)

up:
	$(ENV_FALLBACK_WARN)
	docker compose --env-file $(ENV_FILE) up --build

build:
	$(ENV_FALLBACK_WARN)
	docker compose --env-file $(ENV_FILE) build

down:
	docker compose down

# Local backend: bind the host BACKEND_PORT (container-free dev loop).
dev-backend:
	$(ENV_FALLBACK_WARN)
	@set -a; . $(ENV_FILE); set +a; \
		venv/bin/uvicorn api_main:app --host 127.0.0.1 --port $$BACKEND_PORT

# Local frontend: Vite on FRONTEND_PORT, with VITE_API_BASE exported so the
# client talks to the configured backend.
dev-frontend:
	$(ENV_FALLBACK_WARN)
	@set -a; . $(ENV_FILE); set +a; \
		cd frontend && npm run dev -- --host 127.0.0.1 --port $$FRONTEND_PORT
