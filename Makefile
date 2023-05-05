.PHONY: help
help:  ## Show the help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
.DEFAULT_GOAL := help

.PHONY: shell
shell: ## Shell developer
	@docker exec -ti api-dev bash

.PHONY: lint
lint: ## lint
	@docker exec -ti api-dev black --check .

.PHONY: prod
prod: ##  Start environment production
	@docker-compose -f docker/production/docker-compose.yaml up --build

.PHONY: dev
dev: ## Build and Start environment developer
	@docker-compose  up --build

.PHONY: up
up: ## Start environment developer
	@docker-compose  up