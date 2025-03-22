.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --doctest-modules

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: pipeline
pipeline: ## Run the pipeline
	@echo "🚀 Running the pipeline"
	@uv run used_car_price_prediction/pipeline/main.py

.PHONY: run-api
api: ## Run only the FastAPI backend
	@echo "🚀 Running FastAPI backend service"
	@uv run uvicorn app.main:app --reload

.PHONY: run-ui
ui: ## Run only the Streamlit frontend
	@echo "🚀 Cleaning dataset to display in Data Analysis section in UI"
	@uv run scripts/clean_dataset.py
	@echo "🚀 Running Streamlit frontend service"
	@uv run streamlit run used_car_price_prediction/ui/main.py

.PHONY: application
application: ## Run the application
	@echo "🚀 Cleaning dataset to display in Data Analysis section in UI"
	@uv run scripts/clean_dataset.py
	@echo "🚀 Trigger preprocessing data pipeline"
	@uv run used_car_price_prediction/pipeline/main.py
	@echo "🚀 Running FastAPI backend service"
	@uv run app/main.py &
	@echo "🚀 Running the Streamlit frontend service"
	@uv run streamlit run used_car_price_prediction/ui/main.py

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
