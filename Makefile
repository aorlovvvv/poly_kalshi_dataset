PYTHON ?= python3

.PHONY: profile analysis verify phase2 all paper
.PHONY: pin copula gmm rl-train
.PHONY: autoresearch run-all check-artifacts

# --- Data Pipeline ---
profile:
	$(PYTHON) -m src.analysis.profile_data

analysis:
	$(PYTHON) -m src.analysis.run_models

verify:
	$(PYTHON) -m src.analysis.verify

# --- Phase 1-2: Statistical Tests ---
phase2:
	$(PYTHON) -m src.analysis.phase2_statistical_tests

# --- RQ1: PIN/VPIN Estimation ---
pin:
	$(PYTHON) -m src.analysis.pin_estimation --markets 30

# --- RQ2: Copula Cross-Platform Dependence ---
copula:
	$(PYTHON) -m src.analysis.copula_dependence

# --- RQ2: GMM/SDF Estimation ---
gmm:
	$(PYTHON) -m src.analysis.gmm_sdf --markets 20

# --- RQ3: Autoresearch (run from src/autoresearch/ directory) ---
autoresearch:
	cd src/autoresearch && $(PYTHON) train.py

# --- RQ3: RL Agent Training ---
rl-train:
	$(PYTHON) -m src.autoresearch.rl_agent --episodes 50 --reward sharpe

# --- Orchestration ---
run-all:
	$(PYTHON) -m src.analysis.run_all --all

check-artifacts:
	$(PYTHON) -m src.analysis.run_all --check

# --- Full Pipeline ---
all: profile analysis verify phase2 pin copula gmm

# PDF compilation
paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=paper paper/paper.tex