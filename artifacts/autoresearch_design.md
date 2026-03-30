# Autoresearch Harness for Prediction Markets Paper
## Adapting Karpathy's Pattern to Research Paper Optimization

### The Three-File Architecture

| Autoresearch (LLM) | Our Project | Role |
|---|---|---|
| `prepare.py` (FIXED) | `src/analysis/evaluate_paper.py` + `verify.py` + raw data | The evaluation harness — cannot be edited by the agent |
| `train.py` (EDITABLE) | `src/analysis/run_models.py` + `phase*.py` + `paper/paper.tex` | Everything the agent iterates on |
| `program.md` (INSTRUCTIONS) | `CLAUDE.md` + this document | Agent's playbook |

### The Fitness Metric: Paper Quality Score (PQS)

The single scalar we optimize. Higher is better. Range: [0, 100].

```
PQS = (
    20 * (n_statistical_tests_passing / n_tests_planned)       # Do the analyses run?
  + 15 * (n_latex_tables_generated / n_tables_expected)        # Are tables produced?
  + 15 * (n_figures_generated / n_figures_expected)            # Are figures produced?
  + 10 * (verify_passes)                                       # No digits in prose, all artifacts exist
  + 15 * (findings_coverage_ratio)                             # Are all claims backed by data?
  + 10 * (avg_statistical_significance)                        # How significant are the findings?
  + 15 * (n_paper_sections_drafted / n_sections_expected)      # Is the paper written?
)
```

**Why this works like val_bpb:**
- It's a single number (comparable across iterations)
- It's computed by the FIXED harness (can't be gamed)
- Lower-quality experiments get lower scores and are reverted
- The harness checks that claims match data (prevents fabrication)
- Each component is verifiable from artifacts on disk

### What the Evaluation Harness Checks (evaluate_paper.py)

This is our `prepare.py` — the agent CANNOT modify it.

1. **Statistical tests**: Reads `artifacts/phase*_results.json`. Counts how many tests produced valid results with p-values.
2. **Tables**: Checks `paper/tables/*.tex` exist and are valid LaTeX.
3. **Figures**: Checks `paper/figures/*.png` exist and are non-empty.
4. **Verification**: Runs `verify.py` — ensures no digits in paper prose, all macros resolve.
5. **Findings coverage**: Parses `artifacts/findings.md` and checks that every `\ref{}` in paper.tex points to an existing table/figure.
6. **Significance**: Reads p-values from results JSON, computes average -log10(p).
7. **Paper completeness**: Checks that Introduction, Data, Methods, Results, Conclusion sections have >100 words each.

Output: `{"pqs": 73.5, "breakdown": {...}, "timestamp": "..."}`

### The Experiment Loop

```
LOOP FOREVER:
  1. Read git state (branch, last commit, current PQS)
  2. Review results.tsv — what worked, what didn't
  3. Form hypothesis: "Adding QQ plots for EVT will improve PQS by ~5 points"
  4. Edit the analysis scripts or paper.tex
  5. git commit -m "experiment: add QQ plots for tail analysis"
  6. Run: python3 -m src.analysis.run_models && python3 -m src.analysis.evaluate_paper
  7. Read PQS from output
  8. If PQS improved → keep commit, update baseline
     If PQS same/worse → git reset --hard HEAD~1
  9. Log to results.tsv: commit | pqs | status | description
  10. Repeat — NEVER STOP
```

### Time Budget

- Each analysis iteration: ~2-3 minutes (DuckDB on 36GB is fast)
- Evaluation: ~10 seconds
- Total per experiment: ~3-4 minutes
- Experiments per hour: ~15-20
- Overnight (8 hours): ~120-160 experiments

### Experiment Ideas (Ordered by Expected PQS Impact)

**High impact (PQS +10-15):**
1. Complete the LaTeX table generation (5 tables = +15 points)
2. Draft the Introduction section (+15 points)
3. Generate all 4 expected figures (+15 points)

**Medium impact (PQS +5-10):**
4. Add QQ plots for EVT validation
5. Add calibration curves for XGBoost
6. Write the Data section with proper macros
7. Add per-market-type stratification tables

**Low impact (PQS +1-5):**
8. Try different GPD thresholds (90% vs 95% vs 99%)
9. Try different XGBoost hyperparameters (depth, n_estimators)
10. Add Ljung-Box test as additional efficiency check
11. Try LASSO instead of XGBoost for interpretability comparison
12. Add confidence intervals to all point estimates

**Moonshot (PQS +15-20 if successful, 0 if not):**
13. Cross-platform event matching and lead-lag analysis
14. Intraday pattern × tail risk interaction analysis
15. Market lifecycle analysis (how tail properties change as resolution approaches)

### Key Constraints (What the Agent CANNOT Do)

1. **Cannot modify evaluate_paper.py** — this is the fixed harness
2. **Cannot modify raw data** — parquet files are immutable
3. **Cannot fabricate results** — all claims must trace to artifacts/*.json
4. **Cannot skip verify.py** — it's part of the PQS computation
5. **Cannot install new packages** — use what's in requirements.txt

### Simplicity Criterion (from Karpathy)

"All else being equal, simpler is better."
- A +2 PQS that adds 200 lines of hacky code? Probably not worth it.
- A +2 PQS from deleting unnecessary analysis? Definitely keep.
- Equal PQS but cleaner paper prose? Keep.
- A +0.5 PQS from a marginal finding? Skip, focus on bigger wins.

### results.tsv Format

```tsv
commit	pqs	status	description
baseline	0.0	keep	initial state - no analyses run
a1b2c3d	23.5	keep	Phase 1 profiling complete
b2c3d4e	45.2	keep	Phase 2 statistical tests running
c3d4e5f	43.1	discard	tried alternative GPD threshold - worse fit
d4e5f6g	52.8	keep	added variance ratio LaTeX table
```

### Implementation Steps

1. Create `src/analysis/evaluate_paper.py` (the fixed harness)
2. Create `results.tsv` with header
3. Update Makefile: `make score` target
4. Run baseline to establish PQS = 0
5. Start the autonomous loop

### Why This Pattern Is Powerful for Research

The autoresearch pattern works because it:
- **Prevents the agent from getting lost** — there's always a clear metric to optimize
- **Makes progress measurable** — you can see the PQS increase over time
- **Enables sleep-time research** — agent runs 100+ experiments overnight
- **Builds on success** — git branching means the baseline only gets better
- **Catches regressions** — any change that hurts the paper is auto-reverted
- **Separates creation from evaluation** — the agent can't cheat the metric
