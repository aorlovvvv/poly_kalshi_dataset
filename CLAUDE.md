# CLAUDE.md — Project Context for Research Paper Development

## Project Overview

**Title:** Information, Tails, and Autonomous Discovery in Prediction Markets: A Cross-Platform Microstructure Study
**Author:** Andrey Orlov
**Target venues:** NeurIPS 2026 (Datasets & Benchmarks), ICAIF 2026, Journal of Financial Data Science
**Unified thesis:** Information arrival process — not venue design — is the primary determinant of prediction market microstructure.
**Research questions:**
- RQ1: How does information arrival shape efficiency + informed trading (PIN/VPIN)?
- RQ2: Do returns follow EVT distributions? Cross-platform tail dependence via copulas?
- RQ3: Can autonomous autoresearch discover non-obvious microstructure regularities?
**Methods:** Martingales, EVT, XGBoost, RL (PPO), Copulas, PIN/VPIN, GMM/SDF (planned)
**Canonical design document:** `artifacts/research_design_v2.md`

## Platform Descriptions

### Kalshi (Centralized, Regulated)
- CFTC-regulated event contract exchange (US-based)
- Binary contracts priced in cents (0–99); yes_price + no_price = 100 always
- Fields: `trade_id`, `ticker` (market ID), `yes_price`, `count` (# contracts), `taker_side` (yes/no), `created_time`
- ~72M trades, Jun 2021 – Nov 2025, ~586K unique markets
- Each contract resolves to $1 (yes) or $0 (no)

### Polymarket (Decentralized, On-Chain)
- Polygon-based prediction market using Conditional Token Framework (CTF)
- Two contract types: CTF Exchange (~265M trades) and NegRisk CTF Exchange (~140M trades)
- Amounts in USDC atomic units (6 decimals, divide by 1e6)
- Trade direction determined by `maker_asset_id`:
  - `maker_asset_id = '0'`: maker pays USDC for outcome tokens → `price = maker_amount / taker_amount`, side = 'buy'
  - `maker_asset_id != '0'`: maker offers tokens for USDC → `price = taker_amount / maker_amount`, side = 'sell'
- Timestamps are NULL in trade records; must be joined to `polymarket_blocks` via `block_number`
- Market ID = the non-'0' asset_id (identifies the conditional token)
- ~404M on-chain trades, Mar 2023 – Jan 2026
- Time overlap with Kalshi: Mar 2023 – Nov 2025 (~32 months)

## Repository Structure

```
poly_kalshi_dataset/
├── config/
│   ├── project.yml          # Top-level config (seed, max_rows, paper metadata)
│   ├── datasets.yml          # 4 dataset definitions with parquet globs
│   └── model_spec.yml        # Column mappings + analysis task list
├── data/
│   ├── raw/
│   │   ├── kalshi/
│   │   │   ├── trades/       # ~7,214 parquet files
│   │   │   └── markets/      # ~769 parquet files
│   │   └── polymarket/
│   │       ├── trades/       # Parquet files
│   │       ├── blocks/       # Block timestamps for joining
│   │       └── markets/      # Market metadata
│   └── processed/            # Output of prepare_data.py (currently empty)
├── src/
│   ├── analysis/
│   │   ├── prepare_data.py        # Raw → processed parquet pipeline
│   │   ├── profile_data.py        # Data catalog generator → artifacts/
│   │   ├── run_models.py          # Full analysis engine → tables/figures/findings
│   │   ├── verify.py              # Deterministic checks (no digits in prose, artifacts exist)
│   │   ├── pin_estimation.py      # RQ1: PIN/VPIN estimation (Easley-O'Hara 1996)
│   │   ├── copula_dependence.py   # RQ2: Copula cross-platform dependence
│   │   └── phase2_statistical_tests.py  # Phase 2: VR, BDS, Roll's spread, GPD, XGBoost
│   ├── autoresearch/
│   │   ├── prepare.py             # FIXED harness (daily Sharpe, walk-forward backtest)
│   │   ├── train.py               # EDITABLE strategy (current: sqrt-3trade-tanh-v37)
│   │   ├── program.md             # Agent instructions (v2)
│   │   ├── rl_agent.py            # PPO agent for position sizing (RQ3)
│   │   └── results.tsv            # 101 experiment results
│   └── agents/
│       └── run.py                 # LangGraph recursive agent loop
├── paper/
│   └── paper.tex             # LaTeX paper (sections + figure/table includes)
├── artifacts/                # Generated outputs (data_catalog, findings, figures, tables)
├── Makefile                  # make profile | analysis | verify | all | paper
├── requirements.txt          # Python deps (duckdb, pandas, polars, matplotlib, langgraph, etc.)
└── .env                      # API keys (ANTHROPIC_API_KEY)
```

## Data Pipeline

### Step 1: Profile (`make profile` → `src/analysis/profile_data.py`)
- Reads raw parquet files via DuckDB
- Produces `artifacts/data_catalog.json` and `artifacts/data_catalog.md`
- Generates `paper/results.tex` with LaTeX macros for row/file counts
- Uses parquet metadata for fast row counts (no full scan)

### Step 2: Prepare Data (`src/analysis/prepare_data.py`)
- Standardizes both platforms' trades into a common schema
- Kalshi: `price = yes_price / 100`, `quantity = count`, `notional = price * count`
- Polymarket: joins blocks for timestamps, derives price from amount ratios, converts to USDC
- Outputs to `data/processed/`: `kalshi_trades.parquet`, `polymarket_trades.parquet`, `kalshi_markets.parquet`, `trades_all.parquet`

### Step 3: Analysis (`make analysis` → `src/analysis/run_models.py`)
- Builds `trades_all` view combining both platforms with correct semantics
- Runs 7 analysis tasks (configured in `config/model_spec.yml`):
  1. **trade_summary** — N trades, N markets, date range, avg/median/std price, total notional
  2. **price_distribution** — Quantiles (p01–p99) by platform
  3. **volume_by_day** — Daily trade count and notional volume time series
  4. **trade_size_distribution** — Quantity quantiles, mean/median ratio (skewness indicator)
  5. **hourly_pattern** — Intraday activity by hour (UTC)
  6. **price_concentration** — Fraction of trades in extreme/confident/tossup/uncertain ranges
  7. **taker_side_analysis** — Buy/sell imbalance by platform
- Outputs: LaTeX tables in `paper/tables/`, PNG figures in `paper/figures/`, findings in `artifacts/findings.md`, macros in `paper/results.tex`

### Step 4: Verify (`make verify` → `src/analysis/verify.py`)
- Checks all required artifacts exist
- Ensures NO digits appear in narrative prose of `paper/paper.tex`
- All numbers must come from LaTeX macros or generated tables

## Agent Loop (`python3 -m src.agents.run`)

LangGraph-based recursive pipeline:
```
START → bootstrap → explore → discover → write → critique → bump
                      ↑                                       │
                      └──── needs_more_analysis ──────────────┤
                                                              │
                            write ←── needs_revision ─────────┤
                                      │
                            verify ←── approved/max ──────────┘
                              │
                             END
```

- **bootstrap**: runs `make profile`
- **explore**: runs `make analysis`
- **discover**: LLM reads computed results, identifies key findings and narrative arc
- **write**: LLM drafts/revises Introduction, Data, Results, Conclusion sections
- **critique**: LLM referee checks claims against actual data, flags overclaims
- **bump**: increment iteration counter
- **verify**: deterministic checks
- Max 5 iterations before forcing verify

## Paper Rules (Non-Negotiables)

1. **No invented results.** All claims must be grounded in `artifacts/findings.md`
2. **No digits in prose.** All numbers via LaTeX macros (`\NKalshiTrades`, `\NPolymarketTrades`, `\NStandardizedTrades`, etc.) or generated tables
3. **Column mappings must match verified data semantics** (see `run_models.py` docstring)
4. **Reference all tables and figures** by label:
   - `Table~\ref{tab:trade_summary}`, `Table~\ref{tab:price_distribution}`, `Table~\ref{tab:price_concentration}`, `Table~\ref{tab:trade_size}`, `Table~\ref{tab:taker_side}`
   - `Figure~\ref{fig:price_dist}`, `Figure~\ref{fig:volume_by_day}`, `Figure~\ref{fig:trade_size}`, `Figure~\ref{fig:hourly}`

## Available LaTeX Macros (generated by pipeline)

- `\NStandardizedTrades` — total combined trades
- `\NKalshiTrades`, `\NPolymarketTrades` — per-platform trade counts
- `\NKalshiMarkets`, `\NPolymarketMarkets` — per-platform unique market counts
- `\RowsKalshiTrades`, `\RowsPolymarketTrades`, etc. — raw row counts from profiler

## Current State

- Raw data: present (~36 GB compressed archive + extracted parquet files)
- Processed data: not yet generated (need to run `make profile` then prepare_data, then `make analysis`)
- Paper: skeleton exists with section structure and table/figure includes, but no narrative prose yet
- Artifacts: directories exist but empty (no findings or data catalog generated yet)

## Next Steps for Research Paper

1. Run `make profile` to generate the data catalog
2. Run `prepare_data.py` to create processed parquet files
3. Run `make analysis` to generate all tables, figures, and findings
4. Iterate on paper prose using the discover → write → critique loop
5. Expand analysis with additional tasks as needed (e.g., calibration analysis, spread analysis, market efficiency tests)
6. Run `make verify` to ensure paper quality standards

## Key Design Decisions

- **DuckDB** for all data processing — handles parquet natively, fast on large datasets, SQL-based
- **No pandas for heavy lifting** — DuckDB does aggregation/joins, pandas only for final table formatting
- **Separation of concerns**: data prep → analysis → paper writing are independent stages
- **Deterministic verification**: no manual checking needed; `verify.py` enforces rules programmatically
- **LangGraph agent loop**: automated iteration with LLM-as-referee for quality control

## Autoresearch (Karpathy's Optimization Harness)

The `autoresearch/` directory contains [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), added as a git submodule. While originally designed for autonomous LLM training optimization, the **pattern** is directly applicable to our research pipeline.

### How Autoresearch Works (Three Files)

1. **`prepare.py`** — The **evaluation harness** (UNTOUCHABLE). Defines data loading, tokenizer, and the scoring function (`evaluate_bpb`). The agent cannot modify this — otherwise it would game the metric rather than improve the model.
2. **`train.py`** — The **only file the agent edits**. Contains model architecture, optimizer, and training loop. All experimentation happens here.
3. **`program.md`** — The **agent's instruction manual**. Tells the LLM agent how to behave, what to try, when to keep vs. revert changes.

### The Optimization Loop

```
LOOP FOREVER:
  1. Read current code state (git branch/commit)
  2. Form hypothesis → edit train.py
  3. git commit the change
  4. Run 5-minute time-boxed experiment → get val_bpb score
  5. If score improved → keep commit (new baseline)
     If score equal/worse → git reset --hard (instant revert)
  6. Log result to results.tsv (commit, val_bpb, memory, status, description)
  7. Repeat
```

Key design principles:
- **Fixed time budget** (5 min) makes experiments directly comparable
- **Single metric** (val_bpb) — lower is better, unambiguous
- **Git-based versioning** — every experiment is a commit, reverts are instant
- **Simplicity criterion** — small improvement + ugly complexity = not worth it; equal result + simpler code = keep
- **Never stop** — the agent runs autonomously until manually interrupted

### Mapping Autoresearch to Our Research Project

| Autoresearch (LLM Training) | Our Project (Prediction Markets) |
|---|---|
| `prepare.py` (fixed harness) | `src/analysis/verify.py` + `prepare_data.py` (fixed evaluation) |
| `train.py` (agent edits) | `src/analysis/run_models.py` + `paper/paper.tex` (agent iterates) |
| `program.md` (instructions) | `CLAUDE.md` + `config/model_spec.yml` (research context) |
| `val_bpb` (fitness metric) | Verification pass + findings coverage + paper quality score |
| 5-min training run | DuckDB analysis run (~minutes on 36GB dataset) |
| `results.tsv` (experiment log) | `artifacts/findings.md` + git history |

### Adaptation Strategy

To apply the autoresearch pattern here:
1. **Define a measurable fitness signal** — e.g., number of analysis tasks passing, verification score, findings count, or a composite quality metric
2. **Constrain what the agent can edit** — analysis scripts and paper prose, NOT the evaluation harness or raw data pipeline
3. **Use git-based experiment tracking** — each analysis iteration is a commit; revert if quality degrades
4. **Time-box experiments** — each analysis + write cycle gets a fixed budget
5. **Log everything** — TSV of iterations with metric, status, description
