# AI-Powered Commercial-Forecast Agent

**Academic-First → Enterprise-Ready**

## Table of Contents

- [AI-Powered Commercial-Forecast Agent](#ai-powered-commercial-forecast-agent)
  - [Table of Contents](#table-of-contents)
  - [Purpose (what this project is)](#purpose-what-this-project-is)
  - [Why it matters](#why-it-matters)
  - [Primary Users \& Decisions](#primary-users--decisions)
  - [Scope \& Non-Goals (for this MVP)](#scope--non-goals-for-this-mvp)
    - [In-scope](#in-scope)
    - [Non-goals (MVP)](#non-goals-mvp)
  - [Success Criteria (2-Week MVP)](#success-criteria-2-week-mvp)
  - [Architectural Overview (MVP)](#architectural-overview-mvp)
  - [Data \& Assumptions (MVP)](#data--assumptions-mvp)
  - [Deliverables](#deliverables)
  - [Constraints \& Guardrails](#constraints--guardrails)
  - [Expansion Path (post-MVP)](#expansion-path-post-mvp)
  - [Glossary (for consistency)](#glossary-for-consistency)
  - [File Ownership (where Claude should write code)](#file-ownership-where-claude-should-write-code)
  - [Acceptance Tests (must pass)](#acceptance-tests-must-pass)
  - [Work Order Template (Claude should use this format)](#work-order-template-claude-should-use-this-format)

## Purpose (what this project is)

An end-to-end agent that turns minimal, auditable inputs into **board-ready commercial forecasts** for pipeline drugs. It produces:

- A **TAM/SAM-based adoption forecast** (Bass diffusion + ML overlay)
- A **price/access simulator** that maps price bands to expected coverage (open/PA/niche) and net revenue (GtN)
- A **Monte-Carlo NPV/IRR** engine with explainable drivers (SHAP)
- A **one-pager export** (charts + assumptions) suitable for senior stakeholders

> **Reference asset:** Verona Pharma's ensifentrine (COPD) as a blueprint for design choices (but MVP uses public/synthetic data only).

## Why it matters

Academic teams need credible commercial read-through **before** paying for large datasets or partner engagements. This agent proves value on public/synthetic data, then **scales cleanly** to enterprise datasets (claims, Rx audits, payer policy feeds) without changing core APIs or file schemas.

---

## Primary Users & Decisions

- **PI / Board / Commercial Lead** — "What is a credible price/access strategy and the NPV range under uncertainty?"
- **HEOR / Market Access** — "Which price corridor maximizes value under realistic payer behavior?"
- **Portfolio/BD** — "What's the upside/risk for partner vs. go-it-alone vs. exit?"

---

## Scope & Non-Goals (for this MVP)

### In-scope

- US market only (initially)
- COPD asset benchmark (ensifentrine) + 1 analog back-test
- Clean, deterministic pipeline: `data → forecast → NPV → one-pager`

### Non-goals (MVP)

- No PHI or real EHR ingestion
- No multi-country hierarchy (Bayesian multi-market comes later)
- No production Spark/Snowflake deployment (local Streamlit + notebooks only)

---

## Success Criteria (2-Week MVP)

- Runs end-to-end locally; small, documented synthetic/public CSVs
- **Adoption forecast + NPV P10/P50/P90** generated
- **Price/access toggles** change adoption & net revenue as expected
- **Explainability:** SHAP plots identify top drivers
- **Exportable one-pager** with charts + assumptions box
- **Back-test seed:** 1 analog launch → report Year-1 MAPE

---

## Architectural Overview (MVP)

1. **Data layer** (`data_raw/` → `data_proc/`): tiny CSVs for epi, analogs, price bands, and payer snippets; loader validates schemas.

2. **Forecast core**:
   - **Bass diffusion** → quarterly adopters
   - **GBDT overlay** (thin) → adjusts adoption with a few features (e.g., price_index, access_tier_idx)

3. **Price/Access simulator**: rule-based mapping + (optionally) a tiny classifier; sets adoption ceiling and GtN%.

4. **Economics**: vectorized NPV/IRR with quarterly discounting; **Monte-Carlo** ≥10k runs over key uncertainties.

5. **Explainability**: SHAP (global + local) → driver bar & waterfall PNGs.

6. **UI & Export**: Streamlit sliders + **pptx** export (Jinja2 template) → "board-ready" one-pager.

---

## Data & Assumptions (MVP)

- **Epi:** US COPD prevalence, exacerbation rate, and **eligible-for-nebulizer** estimate (public/synthetic).
- **Analogs:** 2–3 respiratory launches with rough p/q priors and price bands.
- **Price bands → Access tier** mapping (OPEN / PA / NICHE) with hard thresholds (doc in code comments).
- **Economics:** list price, GtN%, COGS%, SG&A ramp, WACC (config in `conf/params.yml`).
- Everything is **documented inline** in notebooks; synthetic data files are prefixed with dates.

---

## Deliverables

- `02_forecast_baseline.ipynb` → adoption + revenue curves
- `03_scenarios_NPV.ipynb` → NPV distribution + driver ranking
- `src/app.py` → Streamlit demo with sliders & **Export one-pager** button
- `reports/board_onepager_v0.1.pptx` → charts + assumptions box
- `reports/tech_notes_v0.2.md` → provenance, model cards, scale path

---

## Constraints & Guardrails

- **Academic-first**: small footprint, pure Python; no paid data required to run MVP.
- **Enterprise-ready**: model APIs stable; later swap in licensed claims/Rx without changing function signatures.
- **No PHI**; only public/synthetic data; keep docstrings and unit tests.
- **Determinism**: set random seeds; ensure figures match between runs given the same config.

---

## Expansion Path (post-MVP)

- Multi-market **Bayesian hierarchical** adoption models
- Payer-policy **LLM parser** for PDFs → JSON, trained classifier for access probability
- **Databricks/Snowflake** deployment; monthly re-forecast jobs; governance reviews
- Larger **analog library** and better instrumented back-testing harness

---

## Glossary (for consistency)

- **TAM/SAM** — Total/Serviceable Addressable Market (eligible patients after inclusion/exclusion rules)
- **Bass p/q** — innovation/imitation coefficients governing adoption
- **GtN%** — Gross-to-Net discount factor (net = list × GtN)
- **Access tier** — OPEN (broad), PA (prior-auth/restricted), NICHE (highly restricted)
- **NPV/IRR** — Discounted cash flow metrics over 10-year horizon (quarterly periods)

---

## File Ownership (where Claude should write code)

- `src/models/bass.py` — Bass functions only (no I/O)
- `src/models/gbdt.py` — Thin ML overlay + SHAP hooks
- `src/access/pricing_sim.py` — Price→access mapping + optional tiny classifier
- `src/econ/npv.py` — Discounting, NPV/IRR, Monte-Carlo
- `src/io/etl.py` — CSV loaders + schema validation
- `src/explain/xai.py` — SHAP plots
- `src/export/onepager.py` — PPT export via Jinja2 & python-pptx
- `src/app.py` — Streamlit UI (sliders, charts, export)

---

## Acceptance Tests (must pass)

- **Bass**: cumulative adoption ≤ market size; monotonic cumulative; edge cases (tiny m, extreme p/q)
- **NPV**: r=0 matches sum of cashflows; constant cashflows match closed-form; quarterly vs annual discount switch
- **Access mapping**: threshold behavior; adoption ceiling applied; GtN updated when tier changes
- **Export**: renders PPTX with non-blank charts & assumptions box

---

## Work Order Template (Claude should use this format)

**Intent**: Implement `<module>` per README

**Plan**:

1. Create/modify files: `<paths>`
2. Add tests: `<tests>`
3. Run locally: commands + expected console output

**Diffs**: show code changes only in the specified files

**Notes**: assumptions, seeds, expected shapes/units

