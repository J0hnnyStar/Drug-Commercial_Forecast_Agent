# Drug-Commercial Forecast Agent (Phase 0)

Production-grade foundation for pharmaceutical commercial forecasting with industry baselines, rigorous statistics, full audit logging, and acceptance gates. Phase 0 validates infrastructure on synthetic data only; Phase 1 will add real launches (N≥50) and backtesting.

## Table of Contents
- [Overview](#overview)
- [Repo Map](#repo-map)
- [Setup](#setup)
- [Core Commands](#core-commands)
- [Acceptance Gates (G1–G5)](#acceptance-gates-g1–g5)

## Overview
Phase 0 delivers:
- Industry baselines: peak sales heuristic, analog projection, patient-flow
- Statistical protocol: temporal split, 5-fold CV on train, Holm–Bonferroni, bootstrap CIs
- Audit & provenance logging: usage, cost, git state, seeds, configs
- Acceptance gates (G1–G5) to prevent over-claims
- CLI and Make targets for reproducible runs

No performance claims on real launches are made in Phase 0. Phase 1 collects real data and runs backtesting plus H1/H2/H3 evaluations.

## Repo Map
- Key modules:
  - `src/data/build_dataset.py`, `src/models/{baselines,analogs,patient_flow}.py`, `src/stats/protocol.py`, `src/utils/audit.py`, `src/cli.py`
  - `evaluation/run_h1.py`, `evaluation/backtesting.py`
  - `reports/complete_conference_paper.tex`, `reports/overleaf/*.md`
  - `AGENTS.md`, `docs/SPEC.md`, `ai_scientist/artifacts/*.json`

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
make supplemental
```

## Core Commands
```bash
python src/cli.py build-data --seed 42
python src/cli.py test-baselines -v
python src/cli.py forecast --method ensemble --years 5
python src/cli.py check_protocol --verbose
python src/cli.py evaluate --seed 42
python src/cli.py gates
python src/cli.py audit
```

## Acceptance Gates (G1–G5)
- **G1 Data**: N≥50 launches, ≥5 therapeutic areas, 5-year revenues; schema-valid parquet; profile published
- **G2 Baselines**: peak heuristic, analogs, patient-flow implemented; tests pass
- **G3 Statistical Rigor**: temporal split, 5-fold CV (train only), Holm–Bonferroni corrections, bootstrap CIs
- **G4 Results**: On held-out test, ensemble beats best single baseline on ≥60% launches; median Y2 APE ≤30%; PI80 in 70–90%
- **G5 Audit**: usage and provenance artifacts present; clean git state

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
- `docs/SPEC.md` — single source of truth for reusable prompt templates
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

## Roadmap
- Phase 1: collect real launches (N≥50, ≥5 TAs); run backtesting; execute H1/H2/H3; update paper Results
- Phase 2: integrate real evidence-grounding sources; fix analog weighting alignment; deploy evaluation CI gates in AI.MED org

