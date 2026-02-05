# A.R.B.O.R. Enterprise - Makefile
.PHONY: help dev dev-up dev-down backend frontend test lint format migrate seed

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Development ----
dev-up: ## Start all dev services (DBs, Redis, Kafka)
	docker-compose -f infrastructure/docker/docker-compose.dev.yml up -d

dev-down: ## Stop all dev services
	docker-compose -f infrastructure/docker/docker-compose.dev.yml down

backend: ## Run backend server
	cd backend && poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend: ## Run frontend dev server
	cd frontend && npm run dev

dev: dev-up backend ## Start everything for development

# ---- Database ----
migrate: ## Run database migrations
	cd backend && poetry run alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="description")
	cd backend && poetry run alembic revision --autogenerate -m "$(MSG)"

seed: ## Seed database with test data
	cd backend && poetry run python -m scripts.seed_data

# ---- Testing ----
test: ## Run all tests
	cd backend && poetry run pytest tests/ -v

test-unit: ## Run unit tests
	cd backend && poetry run pytest tests/unit/ -v

test-integration: ## Run integration tests
	cd backend && poetry run pytest tests/integration/ -v

test-e2e: ## Run e2e tests
	cd backend && poetry run pytest tests/e2e/ -v

test-cov: ## Run tests with coverage
	cd backend && poetry run pytest tests/ --cov=app --cov-report=html

# ---- Code Quality ----
lint: ## Run linters
	cd backend && poetry run ruff check app/
	cd frontend && npm run lint

format: ## Format code
	cd backend && poetry run black app/ tests/
	cd backend && poetry run ruff check --fix app/
	cd frontend && npx prettier --write "**/*.{ts,tsx,js,jsx}"

type-check: ## Run type checks
	cd backend && poetry run mypy app/

# ---- Docker ----
build: ## Build production Docker images
	docker-compose -f infrastructure/docker/docker-compose.prod.yml build

prod-up: ## Start production stack
	docker-compose -f infrastructure/docker/docker-compose.prod.yml up -d

prod-down: ## Stop production stack
	docker-compose -f infrastructure/docker/docker-compose.prod.yml down

# ---- Infrastructure ----
tf-init: ## Initialize Terraform
	cd infrastructure/terraform && terraform init

tf-plan: ## Plan Terraform changes
	cd infrastructure/terraform && terraform plan

tf-apply: ## Apply Terraform changes
	cd infrastructure/terraform && terraform apply
