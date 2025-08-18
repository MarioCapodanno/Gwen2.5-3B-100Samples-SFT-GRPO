PYTHON ?= python
PIP ?= pip

# Defaults for CLI parameters (override on the command line)
MODEL ?= Qwen/Qwen2.5-3B
DATASET ?= your_org/your_dataset
BASE ?= Qwen/Qwen2.5-3B
LORA ?= lora-out
OUTDIR ?= outputs/export
MODEL_NAME ?= qwen2_5_3b_ocr_100s
GGUF ?= $(OUTDIR)/model.gguf

.PHONY: help install install-all install-train install-eda install-eval install-export install-ollama install-dev \
        nb-export eda train eval export ollama \
        format lint test clean build publish

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Available targets:\n"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install package in editable mode
	$(PIP) install -U pip
	$(PIP) install -e .

install-all: ## Install all requirement sets
	$(PIP) install -r requirements/nlp_eda.txt
	$(PIP) install -r requirements/train.txt
	$(PIP) install -r requirements/eval.txt
	$(PIP) install -r requirements/convert_export.txt
	$(PIP) install -r requirements/ollama.txt

install-train: ## Install training requirements
	$(PIP) install -r requirements/train.txt

install-eda: ## Install EDA requirements
	$(PIP) install -r requirements/nlp_eda.txt

install-eval: ## Install evaluation requirements
	$(PIP) install -r requirements/eval.txt

install-export: ## Install merge/convert (GGUF) requirements
	$(PIP) install -r requirements/convert_export.txt

install-ollama: ## Install minimal Ollama-related requirements
	$(PIP) install -r requirements/ollama.txt

install-dev: ## Install dev tools (black, isort, ruff, pytest, build, twine, nbstripout)
	$(PIP) install black isort ruff pytest build twine nbstripout

nb-export: ## Export notebooks into Python modules under gwen25/notebooks
	$(PYTHON) scripts/export_notebooks.py

eda: ## Run EDA pipeline
	gwen25 eda --dataset $(DATASET)

train: ## Run training pipeline
	gwen25 train --model $(MODEL) --dataset $(DATASET)

eval: ## Run evaluation pipeline
	gwen25 eval --model $(MODEL)

export: ## Merge LoRA and convert to GGUF
	gwen25 export --base_model $(BASE) --lora_path $(LORA) --outdir $(OUTDIR)

ollama: ## Create Ollama model from GGUF
	gwen25 ollama --model_name $(MODEL_NAME) --gguf_path $(GGUF)

format: ## Format code with black and isort
	black gwen25 scripts
	isort gwen25 scripts

lint: ## Lint code with ruff
	ruff check gwen25 scripts

test: ## Run tests (if present)
	pytest -q || true

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

build: clean ## Build sdist and wheel under dist/
	$(PYTHON) -m build

publish: ## Upload dist/* using twine (configure credentials first)
	$(PYTHON) -m twine upload dist/*


