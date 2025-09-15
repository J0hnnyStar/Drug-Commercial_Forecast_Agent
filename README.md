# Drug-Commercial Forecast Agent (Phase 5 - Historical Validation)

Production-grade AI-powered pharmaceutical commercial forecasting system with multi-agent architecture, real data validation, and comprehensive audit trails. Currently in Phase 5: validating multi-agent system against historical drug performance and continuously improving forecast accuracy.

## Table of Contents
- [Overview](#overview)
- [Repo Map](#repo-map)
- [Setup](#setup)
- [Core Commands](#core-commands)
- [Acceptance Gates (G1–G5)](#acceptance-gates-g1–g5)

## Overview
**Current Status: Phase 2-5 Implementation & Historical Validation**

Following the MASSIVE_OVERHAUL_PLAN.md timeline:

**Completed Phases**:
- ✅ **Phase 0**: Infrastructure foundation - Real datasets, industry baselines, audit logging, CLI
- ✅ **Phase 1**: Real data collection - SEC filings extraction, drug revenue data pipeline  
- ✅ **Phase 2**: Multi-agent architecture - GPT-5 orchestrator with 4 specialized agents (DeepSeek, Perplexity, Claude, Gemini)

**In Progress**:
- 🔄 **Phase 3**: Experimental design - Statistical framework implementation
- 🔄 **Phase 4**: Implementation pipeline - System monitoring and audit trails
- 🔄 **Phase 5**: Historical validation - Testing against real drug launches (2015-2020)

**Current Focus**: Historical validation against actual drug performance, calibrating multi-agent system parameters, and achieving target accuracy (±25% MAPE vs ±40% industry consultant baseline)

## Repo Map
**Core AI System**:
- `ai_scientist/gpt5_orchestrator.py` - Multi-agent coordinator with 8-step forecasting pipeline
- `ai_scientist/specialized_agents.py` - DeepSeek, Perplexity, Claude, Gemini agent implementations 
- `ai_scientist/system_monitor.py` - Full audit trails and decision tracking (Phase 4)
- `ai_scientist/model_router.py` - LLM provider routing and cost tracking

**Data & Validation**:
- `src/data/fixed_sec_extractor.py` - Corrected SEC revenue data extraction
- `phase5_real_validation.py` - Historical validation runner with real drug data
- `src/models/baselines.py` - Industry baseline methods (peak heuristic, ensemble)
- `results/phase5_real_validation.json` - Latest validation results

**Evaluation & Monitoring**:
- `evaluation/{run_h1,run_h2,run_h3}.py` - Hypothesis testing framework
- `reports/complete_conference_paper.tex` - Updated conference paper with Phase 2-5 results

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
make supplemental
```

## Core Commands

**Phase 5 Validation & Testing**:
```bash
# Run historical validation with real drug data
python phase5_real_validation.py

# Test individual components
python test_calibration.py  # Quick calibration verification
python test_phase2_multiagent.py  # Multi-agent system testing

# Evaluate hypotheses
python evaluation/run_h1.py  # Evidence grounding
python evaluation/run_h2.py  # Temporal stability  
python evaluation/run_h3.py  # Competitive forecasting
```

**Legacy CLI (Phase 0-1)**:
```bash
python src/cli.py build-data --seed 42
python src/cli.py test-baselines -v
python src/cli.py forecast --method ensemble --years 5
python src/cli.py gates
python src/cli.py audit
```

## Validation Status & Acceptance Gates

**Current Performance (Historical Validation)**:
- Multi-agent system: 81.3% MAPE (targeting ≤25% to beat consultant baseline)
- Peak heuristic: 67.3% MAPE (best performing baseline)
- Successfully generating different forecasts per drug (fixed identical prediction issue)
- Keytruda forecast: $5.2B vs $25B actual (underestimate by 5x)
- Repatha forecast: $7.0B vs $1.5B actual (overestimate by 4.7x)

**Acceptance Gates Progress (per MASSIVE_OVERHAUL_PLAN.md)**:
- ✅ **Phase 0 Gates**: Real datasets (N≥50), industry baselines implemented, audit logging
- ✅ **Phase 1 Gates**: Data collection pipeline operational (SEC, FDA sources)
- ✅ **Phase 2 Gates**: Multi-agent architecture with GPT-5 orchestrator + 4 specialized agents
- 🔄 **Phase 3 Gates**: Statistical framework (cross-validation, multiple comparisons)
- 🔄 **Phase 4 Gates**: System monitoring and reproducibility audit trails
- 🔄 **Phase 5 Gates**: Beat consultant baseline (±40% → ±25% MAPE target)

**Critical Success Metrics** (per MASSIVE_OVERHAUL_PLAN Phase 7):
- 🔄 Beat baseline methods on held-out test set
- 🔄 Achieve ±25% accuracy vs ±40% industry consultant baseline
- 🔄 Proper statistical validation with real drug launches
- 🔄 Case studies on blockbusters (Keytruda, Humira) and failures

## Data & Schemas
Phase 0 uses a synthetic generator to validate the pipeline. Phase 1 introduces real launches.

- `data_proc/launches.parquet` (one row per launch):
  - `launch_id`, `drug_name`, `company`, `approval_date`, `indication`, `mechanism`, `route`, `therapeutic_area`
  - `eligible_patients_at_launch`, `market_size_source`, `list_price_month_usd_launch`, `net_gtn_pct_launch`
  - `access_tier_at_launch` (OPEN|PA|NICHE), `price_source`, `competitor_count_at_launch`
  - `clinical_efficacy_proxy` (0–1), `safety_black_box` (bool), `source_urls` (json)
- `data_proc/launch_revenues.parquet` (one row per launch-year):
  - `launch_id`, `year_since_launch` (0..4), `revenue_usd`, `source_url`
- `data_proc/analogs.parquet` (optional):
  - `launch_id`, `analog_launch_id`, `similarity_score` (0–1), `justification`

## Evaluation & Backtesting
- H1 (Evidence Grounding): `evaluation/run_h1.py` compares heuristic vs analog vs ensemble (with intervals). Real “grounding” requires external sources (Phase 1).
- Backtesting: `evaluation/backtesting.py` runs rolling quarterly windows; aggregates Y2 APE, peak APE, trajectory MAPE; writes `results/backtest_results.json` and checks G4.

## Audit & Reproducibility
- `results/usage_log.jsonl` — API usage (provider, model, tokens, cost, prompt/response hashes)
- `results/run_provenance.json` — git commit, branch, dirty flag; seed, config snapshot; runs summary
- All CLIs accept `--seed` where applicable for deterministic behavior

## Writing & Paper
- `reports/complete_conference_paper.tex` — aligned to Phase 0 (no over-claims)
- Agent-updated sections: `reports/overleaf/INTRODUCTION.md`, `METHODS.md`, `RESULTS.md`

## Prompts & Agents Policy
- `AGENTS.md` — roles, gates (G1–G5), execution protocol
- `docs/specs/SPEC.md` — single source of truth for reusable prompt templates
- Policy: Agents MUST use SPEC templates; propose changes by versioning SPEC, not ad-hoc prompts

## Contributing
- Open PRs to `pi-review` branch; CI will check gates artifacts
- Commit configs, code, prompts, specs, small protocol JSONs; avoid secrets/large raw dumps
- Code style: clear names, small functions, behavior-focused tests

## Troubleshooting
- Dataset missing: run `python src/cli.py build-data --seed 42`
- Baseline tests failing: `python -m pytest tests/test_baselines.py -v`
- Audit shows git dirty: commit or stash changes before runs for G5
- Windows paths: use PowerShell activation `.venv\Scripts\Activate.ps1`

## MASSIVE_OVERHAUL_PLAN Progress

**Timeline (per MASSIVE_OVERHAUL_PLAN.md)**:

**Completed Phases**:
- ✅ **Phase 0 (Week 0-1)**: Reality check, real datasets, industry baselines, audit hooks, CLI
- ✅ **Phase 1 (Week 1-2)**: Real-world data collection agent (SEC filings, FDA approvals)  
- ✅ **Phase 2 (Week 2-3)**: Multi-agent system with GPT-5 orchestrator + specialized agents

**In Progress (Week 3-6)**:
- 🔄 **Phase 3 (Week 3-4)**: Experimental design - Statistical framework with proper validation
- 🔄 **Phase 4 (Week 4-5)**: Implementation pipeline - Monitoring, audit trails, reproducibility  
- 🔄 **Phase 5 (Week 5-6)**: Historical validation - Testing on drugs launched 2015-2020

**Future Phases**:
- **Phase 6 (Week 6-7)**: Paper rewrite - "Multi-Agent System for Pharmaceutical Commercial Forecasting: Validation on 100 Real Drug Launches"
- **Phase 7 (Week 7-8)**: Conference submission with real validation results

**Success Target**: "Beat industry consultant baseline (±20% accuracy on real drug launches)" - Currently at 81.3% MAPE, targeting ≤25% MAPE to achieve consultant-level performance

