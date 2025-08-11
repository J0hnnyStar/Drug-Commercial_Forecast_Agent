# AI-Powered Commercial-Forecast Agent (MVP, 2-Week Plan)

**Goal (2 weeks):** Ship a clickable demo that runs end-to-end (data → forecast → NPV → exportable 1-pager) and a tight technical note. Academic-first, enterprise-ready.

## 0) Outcomes & Acceptance Criteria

### Week-1 DoD (Definition of Done)

- [ ] Reproducible repo; `make setup && make run-notebooks` works
- [ ] `02_forecast_baseline.ipynb` → quarterly revenue curve
- [ ] `03_scenarios_NPV.ipynb` → NPV distribution (P10/P50/P90)
- [ ] `reports/board_onepager_v0.1.pptx` exported from notebook

### Week-2 DoD

- [ ] Price/Access simulator toggles adoption ceilings & net revenue
- [ ] SHAP plots for drivers included in 1-pager
- [ ] Streamlit micro-app with sliders + "Export 1-pager"
- [ ] 1 analog back-test (Year-1 MAPE reported)
- [ ] `TECH_NOTES_v0.2.md` (governance, data provenance, scale path)

## 1) Repository Layout (authoritative)

```
forecast-agent/
  conf/
    params.yml                 # central assumptions & toggles
  data_raw/                    # input CSVs (versioned by date prefix)
  data_proc/                   # cleaned snapshots (do not edit by hand)
  notebooks/
    01_scope_and_assumptions.ipynb
    02_forecast_baseline.ipynb
    03_scenarios_NPV.ipynb
  src/
    io/etl.py
    models/bass.py
    models/gbdt.py
    access/pricing_sim.py
    econ/npv.py
    explain/xai.py
    app.py                     # Streamlit micro-app (Week-2)
    export/onepager.py
  reports/
    board_onepager_v0.1.pptx
    tech_notes_v0.2.md
  tests/
    test_bass.py
    test_npv.py
    test_access_rules.py
  Makefile
  README.md  <-- (this file)
  requirements.txt
```

**Cursor/Claude rule:** Do not move files across folders unless instructed. Add new functions with tests.

## 2) Environment & Commands

**Python:** 3.11 (>=3.10 OK)

### Install:

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### requirements.txt (minimal to start):

```txt
pandas
numpy
scikit-learn
xgboost
statsmodels
scipy
matplotlib
shap
pyyaml
python-pptx
jinja2
streamlit
```

*(Add pyspark + spark-nlp later when EHR text enters.)*

### Make targets:

```makefile
setup:        ## create venv & install
    python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run-notebooks:## run core notebooks headless (optional)
    echo "Open notebooks/01_*.ipynb in Cursor and run interactively"

app:          ## run Streamlit app
    streamlit run src/app.py

test:         ## run tests
    pytest -q
```
## 3) Data: "Starter Pack" (Week-1)

Put light, public tables in `data_raw/` to prove the pipeline:

- `epi_us_copd.csv` → columns: year, prevalence, exacerbation_rate, eligible_neb_estimate
- `analogs_resp.csv` → drug, launch_qtr, p_init, q_init, price_band, access_tier
- `price_assumptions.csv` → list_price_month, gtn_pct, access_tier
- `payer_rules_snippets.csv` → payer, rule_text, implied_tier (OPEN/PA/NICHE)

**Claude task:** If a file is missing, generate a synthetic but plausible CSV with documented assumptions and save to `data_raw/` (and mention it in the notebook cell).

## 4) Modeling Scope (MVP)

### 4.1 Bass Diffusion (core spine)

- **Inputs:** market size m (eligible patients), p, q (from analogs/priors)
- **Output:** quarterly adopters; cumulative adoption curve

### 4.2 GBDT overlay (thin layer)

- **Model:** XGBoost regressor to adjust Bass output ± based on: price_index, access_tier_idx, specialist_density_proxy, marketing_proxy
- **Guardrail:** keep feature count small; avoid overfit; log SHAP importances

### 4.3 Price & Access Simulator v0.1

- **Rule-first mapping:** price band → access tier (OPEN / PA / NICHE)
- **Optional classifier:** XGBoost (binary) "favorable_at_P?" trained on hand-labeled ~30 rows
- **Effects:** access tier limits adoption ceiling and sets gtn_pct

### 4.4 NPV & Monte-Carlo

- **Horizon:** 10 years, quarterly discounting
- **Params:** WACC, COGS%, SG&A ramp (declining), GtN %
- **Monte-Carlo:** ≥10k runs over p, q, price, gtn, adherence → P10/P50/P90 NPV

### 4.5 Explainability

- SHAP on GBDT overlay; save top-driver bar and local waterfall

### 4.6 Export

- `src/export/onepager.py` → Jinja2 + python-pptx template
- **Content:** TAM/SAM, adoption curve, NPV fan, 3 key levers, assumptions box

## 5) Week-by-Week Plan (with AI-friendly tasks)

### Week-1 (Days 1–5): Minimal Spine

#### Repo & config
- Create `conf/params.yml` with defaults (see sample below)
- ✅ **Claude:** generate schema-validated loader in `io/etl.py`

#### Data pack & assumptions
- Fill `data_raw/*` (real or synthetic), document in `01_scope_and_assumptions.ipynb`

#### Bass + Revenue
- Implement `models/bass.py` and baseline chart in `02_forecast_baseline.ipynb`
- ✅ **Claude:** add `tests/test_bass.py` with edge cases (tiny m, extreme p/q)

#### NPV + Monte-Carlo
- Implement `econ/npv.py` and MC in `03_scenarios_NPV.ipynb`
- ✅ **Claude:** add `tests/test_npv.py` (vectorized correctness, discount math)

#### Export 1-pager v0.1
- `src/export/onepager.py` and template in `/reports`
- Snapshot charts from notebooks

### Week-2 (Days 6–10): Credibility & Demo

#### Price/Access simulator v0.1
- `access/pricing_sim.py` rule table + optional tiny classifier
- ✅ **Claude:** `tests/test_access_rules.py` for mapping consistency

#### Explainability
- `explain/xai.py` → SHAP bar & waterfall; embed in 1-pager

#### Streamlit micro-app
`src/app.py`:
- **Tab "Forecast":** sliders (price, access tier, marketing), charts (adoption, revenue)
- **Tab "NPV":** sliders (WACC, GtN, SG&A), NPV histogram + P10/P50/P90
- **Button:** "Export 1-pager"

#### Ensifentrine mini-benchmark & analog back-test
- Add `reports/benchmark_sheet.csv` (inputs/outputs used)
- One analog back-test → report Year-1 MAPE in `tech_notes_v0.2.md`

#### Polish & handoff
- Update `TECH_NOTES_v0.2.md` (provenance, PHI statement, model cards, scale path)
- Dry-run demo

## 6) Sample Config (conf/params.yml)

```yaml
market:
  geography: US
  eligible_patients: 1200000      # neb-eligible COPD (proxy)
bass:
  p: 0.03
  q: 0.40
gbdt_overlay:
  use: true
  features: [price_index, access_tier_idx, specialist_density_proxy, marketing_proxy]
price_access:
  list_price_month_usd: 2950
  gtn_pct: 0.65                   # net = list * gtn_pct
  access_tier: PA                 # OPEN | PA | NICHE
economics:
  wacc_annual: 0.10
  cogs_pct: 0.15
  sga_launch_annual: 350000000
  sga_decay_to_pct: 0.5
simulation:
  periods_quarterly: 40
  monte_carlo_runs: 10000
  p_sd: 0.01
  q_sd: 0.05
  gtn_sd: 0.05
```
## 7) Minimal Code Stubs (drop-in)

### src/models/bass.py

```python
import numpy as np

def bass_adopters(T: int, m: float, p: float, q: float) -> np.ndarray:
    """Quarterly adopters per period."""
    N = np.zeros(T, dtype=float); cum = 0.0
    for t in range(T):
        f = (p + q*(cum/m)) * (1 - cum/m)
        N[t] = max(0.0, m * f)
        cum += N[t]
    return N
```

### src/econ/npv.py

```python
import numpy as np

def discount_factors(r_quarterly: float, T: int) -> np.ndarray:
    return np.array([(1 + r_quarterly) ** t for t in range(T)], dtype=float)

def npv(cashflows: np.ndarray, r_annual: float, quarterly=True) -> float:
    r = r_annual/4 if quarterly else r_annual
    disc = discount_factors(r, len(cashflows))
    return float((cashflows / disc).sum())
```

### src/access/pricing_sim.py

```python
ACCESS_TIERS = {"OPEN": 1.0, "PA": 0.6, "NICHE": 0.25}  # adoption ceilings

def tier_from_price(list_price_month: float) -> str:
    if list_price_month < 800:   return "OPEN"
    if list_price_month < 1800:  return "PA"
    return "NICHE"
```
## 8) Streamlit Spec (Week-2)

### Tab: Forecast
- **Inputs:** list_price_month, access_tier, marketing_proxy
- **Outputs:** adoption curve (line), revenue (bar)

### Tab: NPV
- **Inputs:** WACC, GtN, SG&A
- **Outputs:** MC histogram, P10/P50/P90 cards

### Export
- **Button** → call `export/onepager.py` to render PPTX with current figures

## 9) Testing Strategy

- `tests/test_bass.py` → mass conservation, monotonic cumulative, stability under extreme p/q
- `tests/test_npv.py` → closed-form checks (r=0, constant cashflows)
- `tests/test_access_rules.py` → price thresholds map to expected tiers and ceilings
- **CI (optional later):** GitHub Actions running `make test` on PR

## 10) Governance & Scale (living doc: reports/tech_notes_v0.2.md)

- **Data provenance table** (source, refresh cadence, license)
- **PHI handling:** MVP uses only public/aggregated, no identifiers
- **Model cards:** inputs, outputs, limits, retrain plan
- **Scale path:** migrate compute to Databricks/Snowflake; add Spark NLP for note mining; connect licensed claims/Rx when available

## 11) Prompts for Cursor + Claude (copypaste as needed)

### A) Architect Prompt (for planning code)

> You are the lead ML architect. Respect the repo structure in README.
> Output: a step plan, file-by-file diffs, and tests to add.
> Do NOT create new folders unless specified. Use small, composable functions.

### B) Coder Prompt (for implementing a module)

> Implement <module> exactly where specified.
> 
> Read conf/params.yml.
> 
> Write pure functions with type hints, no I/O in model code.
> 
> Add/Update unit tests under tests/.
> 
> Run a tiny demo snippet in the docstring showing expected shapes/units.
> If a dependency is missing, add to requirements.txt and explain why.

### C) Reviewer Prompt (for PR review)

> Review diffs for correctness, complexity, and test coverage.
> Check edge cases: extreme p/q, zero market, negative prices.
> Ensure plots and PPT export functions produce deterministic outputs given seed.

### D) Data Steward Prompt (when generating synthetic CSVs)

> Create plausible synthetic data matching the schemas in README.
> Keep ranges realistic; document all assumptions in the notebook cell above the load.
> Save to data_raw/ with a date prefix; do not overwrite existing files.

## 12) Task Board (checklist)

- [ ] Create `conf/params.yml` & loader
- [ ] Fill `data_raw/*` (synthetic allowed)
- [ ] Implement Bass + tests
- [ ] Implement NPV + MC + tests
- [ ] Baseline charts + 1-pager v0.1
- [ ] Price/Access simulator + tests
- [ ] SHAP driver plots
- [ ] Streamlit app + Export button
- [ ] Analog back-test (MAPE)
- [ ] `TECH_NOTES_v0.2.md`

## 13) Notes for Future Enterprise Mode

- Swap synthetic/public data with licensed claims/Rx; keep same schemas
- Add payer-policy LLM parser (PDF → JSON) and trained classifier
- Introduce Bayesian hierarchical adoption model for multi-market launches
- Move jobs to pipelines; schedule monthly re-forecast; governance reviews