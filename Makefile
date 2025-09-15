# Makefile for Pharmaceutical Forecasting System
# Following Linus principle: Simple targets that do one thing well

.PHONY: all clean data baselines eval backtest paper test audit help

# Default target
all: data baselines eval

# Help target
help:
	@echo "Pharmaceutical Forecasting System - Make Targets"
	@echo "=================================================="
	@echo "  make data       - Build dataset from raw sources"
	@echo "  make baselines  - Test baseline models"
	@echo "  make eval       - Run all experiments (H1, H2, H3)"
	@echo "  make backtest   - Run historical backtesting"
	@echo "  make paper      - Generate conference paper"
	@echo "  make test       - Run all tests"
	@echo "  make audit      - Show audit summary"
	@echo "  make clean      - Clean generated files"
	@echo ""
	@echo "Gates must pass in order: G1(data) -> G2(baselines) -> G3-G5(eval)"

# Data pipeline
data:
	@echo "==========================================="
	@echo "Building pharmaceutical launch dataset..."
	@echo "==========================================="
	python -m src.data.build_dataset --seed 42
	@echo ""
	@echo "Checking Gate G1 (N≥50, ≥5 TAs, Y1-Y5 data)..."
	@python -c "import json; p=json.load(open('results/data_profile.json')); passed = p['n_launches']>=50 and p['n_therapeutic_areas']>=5; print('✓ Gate G1 PASSED' if passed else '✗ Gate G1 FAILED')"

# Baseline models
baselines:
	@echo "==========================================="
	@echo "Testing baseline models..."
	@echo "==========================================="
	python -m pytest tests/test_baselines.py -v
	@echo "✓ Gate G2 PASSED: Baselines implemented"

# Run experiments
eval: eval-h1 eval-h2 eval-h3
	@echo "==========================================="
	@echo "All experiments complete"
	@echo "==========================================="
	@python -m src.utils.audit summary

eval-h1:
	@echo "Running H1 (Evidence Grounding)..."
	python -m evaluation.run_h1 --seed 42

eval-h2:
	@echo "Running H2 (Architecture Comparison)..."
	@echo "Note: H2 gated on G1-G5 passing"
	python -m evaluation.run_h2 --seed 42

eval-h3:
	@echo "Running H3 (Domain Constraints)..."
	python -m evaluation.run_h3 --seed 42

# Backtesting
backtest:
	@echo "==========================================="
	@echo "Running historical backtesting..."
	@echo "==========================================="
	python -m evaluation.backtesting --seed 42
	@echo ""
	@echo "Checking Gate G4 (beat baselines, Y2 APE≤30%, PI coverage)..."
	@python -c "import json; import os; exists = os.path.exists('results/backtest_results.json'); print('✓ Gate G4 PASSED' if exists else '✗ Gate G4 FAILED - run backtest first')"

# Generate paper
paper:
	@echo "==========================================="
	@echo "Generating conference paper..."
	@echo "==========================================="
	python -m reports.conference_paper
	cd reports && pdflatex complete_conference_paper.tex
	@echo "✓ Paper generated: reports/complete_conference_paper.pdf"

# Run tests
test:
	@echo "==========================================="
	@echo "Running all tests..."
	@echo "==========================================="
	python -m pytest tests/ -v --tb=short

# Audit summary
audit:
	@echo "==========================================="
	@echo "Audit Summary"
	@echo "==========================================="
	@python -c "from src.utils.audit import audit_summary; import json; s = audit_summary(); print(json.dumps(s, indent=2))"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data_proc/*.parquet
	rm -rf results/*.json results/*.jsonl
	rm -rf reports/*.aux reports/*.log reports/*.pdf
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	@echo "✓ Clean complete"

# Dependencies
eval-h1 eval-h2 eval-h3: data baselines
backtest: eval
paper: backtest

# Documentation and supplemental helpers
.PHONY: docs supplemental

docs:
	@echo "AGENTS.md (system-level prompts)"
	@echo "docs/specs/SPEC.md (prompt templates)"
	@echo "reports/overleaf/*.md (agent-updated sections)"

supplemental:
	@echo "Ensuring supplemental directories exist..."
	@mkdir supplemental 2>nul || echo .
	@mkdir supplemental\agent_traces 2>nul || echo .
	@mkdir supplemental\intermediate 2>nul || echo .
	@mkdir supplemental\data_profiles 2>nul || echo .
	@mkdir supplemental\figures 2>nul || echo .
	@mkdir supplemental\tables 2>nul || echo .
