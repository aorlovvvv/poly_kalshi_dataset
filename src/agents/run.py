"""
LangGraph recursive agent loop for prediction-market research paper.

Graph structure:
  START → bootstrap → explore → discover → write → critique → bump
                        ↑                                       │
                        └──── (needs_more_analysis) ────────────┤
                                                                │
                              write ←── (needs_revision) ───────┤
                                                                │
                              verify ←── (approved/max) ────────┘
                                │
                               END

Phases:
  bootstrap: run `make profile` to produce data catalog
  explore:   run `make analysis` to produce tables/figures/findings
  discover:  LLM reads computed results, identifies key findings and story
  write:     LLM drafts/revises paper sections grounded in findings
  critique:  LLM referee evaluates paper against actual data
  verify:    deterministic checks (artifacts exist, no digits in prose)
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """\
You are a senior ML research scientist writing a comparative microstructure study of \
Polymarket (decentralized, on-chain) and Kalshi (centralized, regulated) prediction markets.

Data reality you must internalize:
- Kalshi trades: yes_price in cents (0-99), yes_price+no_price=100 always. count = contracts. \
  taker_side = yes/no. ~72M trades over Jun 2021 - Nov 2025, ~586K markets, all binary.
- Polymarket trades: ~404M on-chain trades, Mar 2023 - Jan 2026. Prices derived from \
  maker_amount/taker_amount (direction depends on maker_asset_id). Two contract types: \
  CTF Exchange (265M) and NegRisk CTF Exchange (140M).
- Kalshi markets: 7.7M snapshots with bid/ask/volume/result. ~76% resolve 'no', ~24% 'yes'.
- Time overlap: Mar 2023 - Nov 2025 (~32 months).

Research orientation: ML/quantitative finance. Think distributions, patterns, statistical \
tests, calibration, prediction, not causal inference. Write for a venue like NeurIPS \
(Datasets & Benchmarks), ICAIF, or Journal of Financial Data Science.\
"""


class GraphState(TypedDict, total=False):
    iteration: int
    max_iterations: int
    key_findings: str
    paper_story: str
    last_critique: str
    last_error: str
    needs_revision: bool
    needs_more_analysis: bool


def sh(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{out}")
    return out


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _llm():
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        temperature=0,
    )


def _ask(system: str, user: str) -> str:
    return _llm().invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]).content


# ═══════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════

def node_bootstrap(state: GraphState) -> dict:
    """Run make profile to produce the data catalog."""
    sh(["make", "profile"])
    print("[bootstrap] Data catalog ready.")
    return {}


def node_explore(state: GraphState) -> dict:
    """Run make analysis to produce tables, figures, and findings digest."""
    try:
        sh(["make", "analysis"])
        print("[explore] Analysis complete.")
        return {"last_error": "", "needs_more_analysis": False}
    except RuntimeError as e:
        msg = str(e)[-2000:]
        print(f"[explore] ERROR: {msg[:200]}")
        return {"last_error": msg, "needs_more_analysis": False}


def node_discover(state: GraphState) -> dict:
    """LLM reads actual computed results and identifies the story."""
    if state.get("last_error"):
        return {
            "key_findings": "",
            "paper_story": "",
            "needs_revision": True,
            "last_critique": f"Analysis pipeline failed:\n{state['last_error'][:1500]}",
        }

    findings = _read(ROOT / "artifacts" / "findings.md")
    catalog = _read(ROOT / "artifacts" / "data_catalog.md")

    prompt = f"""You have just run a comprehensive analysis of Polymarket vs Kalshi trade data.
Below are the ACTUAL COMPUTED RESULTS — real numbers from the data. Your job is to:

1. Identify the 3-5 most interesting/surprising/publishable findings.
2. For each finding, explain WHY it matters for the ML/finance community.
3. Propose a coherent narrative arc (the "story" of the paper).
4. Note any findings that need deeper investigation in a follow-up analysis.

Be specific. Cite actual numbers from the results. Do not invent or speculate
beyond what the numbers show.

DATA CATALOG:
{catalog}

COMPUTED FINDINGS:
{findings}

Output your response in this format:
KEY FINDINGS:
1. [finding with specific numbers]
2. ...

PAPER STORY:
[2-3 paragraph narrative arc]

FOLLOW-UP NEEDED:
[any additional analyses worth running, or "none"]
"""
    resp = _ask(SYSTEM_PROMPT, prompt)

    return {
        "key_findings": resp,
        "needs_revision": True,
        "needs_more_analysis": False,
    }


def node_write(state: GraphState) -> dict:
    """LLM writes/revises paper sections grounded in actual findings."""
    findings_text = state.get("key_findings", "")
    critique = state.get("last_critique", "")
    catalog = _read(ROOT / "artifacts" / "data_catalog.md")
    raw_findings = _read(ROOT / "artifacts" / "findings.md")
    current_paper = _read(ROOT / "paper" / "paper.tex")

    prompt = f"""Write the paper sections for a prediction-market microstructure study.

You must write grounded, specific prose based on REAL computed findings below.
Every numeric claim must use a LaTeX macro (e.g., $\\NKalshiTrades$, $\\NPolymarketTrades$,
$\\NStandardizedTrades$, $\\NKalshiMarkets$, $\\NPolymarketMarkets$) — do NOT write
any digit in the narrative text.

You should reference tables and figures by label:
  Table~\\ref{{tab:trade_summary}}, Table~\\ref{{tab:price_distribution}},
  Table~\\ref{{tab:price_concentration}}, Table~\\ref{{tab:trade_size}},
  Figure~\\ref{{fig:price_dist}}, Figure~\\ref{{fig:volume_by_day}},
  Figure~\\ref{{fig:trade_size}}, Figure~\\ref{{fig:hourly}}

KEY FINDINGS AND STORY:
{findings_text}

RAW COMPUTED RESULTS:
{raw_findings}

PREVIOUS CRITIQUE TO ADDRESS:
{critique}

CURRENT PAPER (for reference):
{current_paper}

Output EXACTLY ONE fenced code block containing the LaTeX for these sections:
\\section{{Introduction}}, \\section{{Data}}, \\section{{Results}}, \\section{{Conclusion}}

Rules:
- NO digits anywhere in the text. Use macros for all numbers.
- NO \\documentclass, \\begin{{document}}, or preamble.
- Reference every table and figure that exists.
- Be specific about what the data shows. Do not hedge excessively.
- Frame contributions in ML/quantitative finance terms.
- Introduction should motivate why cross-platform comparison matters.
- Data section should describe both platforms' mechanics clearly.
- Results should walk through each analysis with interpretation.
- Conclusion should summarize findings and suggest future work.
"""
    resp = _ask(SYSTEM_PROMPT, prompt)

    # Extract the code block
    blocks = []
    cur: list[str] = []
    in_block = False
    for line in resp.splitlines():
        if line.strip().startswith("```"):
            if in_block:
                blocks.append("\n".join(cur).strip())
                cur = []
                in_block = False
            else:
                in_block = True
            continue
        if in_block:
            cur.append(line)

    if not blocks:
        return {"needs_revision": True, "last_critique": "Writer did not produce a code block."}

    latex_content = blocks[0]

    paper_path = ROOT / "paper" / "paper.tex"
    paper = paper_path.read_text(encoding="utf-8")

    def replace_section(src: str, section: str, new_body: str) -> str:
        pattern = rf"(\\section\{{{section}\}})(.*?)(?=\\section\{{|\\end\{{document\}})"
        m = re.search(pattern, src, flags=re.S)
        if not m:
            return src
        return src[:m.start(2)] + "\n" + new_body.strip() + "\n\n" + src[m.end(2):]

    for section_name in ["Introduction", "Data", "Results", "Conclusion"]:
        section_match = re.search(
            rf"\\section\{{{section_name}\}}(.*?)(?=\\section\{{|$)",
            latex_content, flags=re.S
        )
        if section_match:
            body = section_match.group(1).strip()
            if body:
                paper = replace_section(paper, section_name, body)

    paper_path.write_text(paper, encoding="utf-8")
    print("[write] Paper sections updated.")

    return {"needs_revision": True}


def node_critique(state: GraphState) -> dict:
    """Senior referee evaluates paper against actual computed results."""
    paper = _read(ROOT / "paper" / "paper.tex")
    findings = _read(ROOT / "artifacts" / "findings.md")
    catalog = _read(ROOT / "artifacts" / "data_catalog.md")

    prompt = f"""You are a senior ML reviewer evaluating a prediction-market microstructure paper.

You have access to the ACTUAL COMPUTED RESULTS. Check the paper against them.

Evaluate:
(a) Are all claims supported by the computed results? Flag any overclaims.
(b) Does the narrative contain any hard-coded digits? (It must not.)
(c) Is the framing ML-appropriate? (Not pure economics jargon.)
(d) Are tables and figures properly referenced?
(e) Is the story coherent? Does the paper have a clear contribution?
(f) Is the writing at NeurIPS/ICAIF quality? Specific, precise, not vague.
(g) Does the paper accurately describe how prices are computed on each platform?

COMPUTED RESULTS (ground truth):
{findings}

DATA CATALOG:
{catalog}

PAPER UNDER REVIEW:
{paper}

Return EXACTLY ONE of:
- "APPROVED" (if the paper is ready for a first draft)
- A critique with at most 8 specific, actionable bullets. For each bullet,
  say exactly what to change and why.
"""
    c = _ask(SYSTEM_PROMPT, prompt).strip()

    if "APPROVED" in c.split("\n")[0] and len(c) < 100:
        print("[critique] APPROVED")
        return {"needs_revision": False, "last_critique": ""}

    print(f"[critique] Revision requested ({c[:100]}...)")
    return {"needs_revision": True, "last_critique": c}


def node_verify(state: GraphState) -> dict:
    """Run deterministic verification checks."""
    sh(["make", "verify"])
    print("[verify] PASSED")
    return {}


def node_bump(state: GraphState) -> dict:
    return {"iteration": state.get("iteration", 0) + 1}


# ═══════════════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════════════

def route_after_critique(state: GraphState) -> Literal["explore", "write", "verify"]:
    it = state.get("iteration", 0)
    max_it = state.get("max_iterations", 5)

    if not state.get("needs_revision", True):
        return "verify"
    if it >= max_it:
        return "verify"
    if state.get("needs_more_analysis"):
        return "explore"
    return "write"


# ═══════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════

def build_app():
    g = StateGraph(GraphState)

    g.add_node("bootstrap", node_bootstrap)
    g.add_node("explore", node_explore)
    g.add_node("discover", node_discover)
    g.add_node("write", node_write)
    g.add_node("critique", node_critique)
    g.add_node("verify", node_verify)
    g.add_node("bump", node_bump)

    g.add_edge(START, "bootstrap")
    g.add_edge("bootstrap", "explore")
    g.add_edge("explore", "discover")
    g.add_edge("discover", "write")
    g.add_edge("write", "critique")
    g.add_edge("critique", "bump")
    g.add_conditional_edges("bump", route_after_critique, {
        "explore": "explore",
        "write": "write",
        "verify": "verify",
    })
    g.add_edge("verify", END)

    return g.compile()


def main():
    app = build_app()
    result = app.invoke({"iteration": 0, "max_iterations": 5, "needs_revision": True})
    print(f"\nFinished after {result.get('iteration', '?')} iterations.")


if __name__ == "__main__":
    main()
