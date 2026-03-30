"""
Unified Analysis Pipeline — Runs all RQ1-RQ3 analyses via subprocess.

Usage:
    python -m src.analysis.run_all [--rq1] [--rq2] [--all] [--check] [--dry-run]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
PYTHON = sys.executable


def _run_module(desc: str, module: str, args: list = None, dry_run: bool = False):
    """Run a Python module as a subprocess."""
    cmd = [PYTHON, "-m", module] + (args or [])
    print(f"\n--- {desc} ---")
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  {status} in {elapsed:.1f}s")


def run_rq1(dry_run: bool = False):
    """RQ1: Information Arrival, Efficiency, and Informed Trading."""
    print("=" * 70)
    print("RQ1: Information Arrival, Efficiency, and Informed Trading")
    print("=" * 70)
    _run_module("Phase 2 Statistical Tests", "src.analysis.phase2_statistical_tests", dry_run=dry_run)
    _run_module("PIN/VPIN Estimation", "src.analysis.pin_estimation", ["--markets", "30"], dry_run=dry_run)


def run_rq2(dry_run: bool = False):
    """RQ2: Tail Risk Characterization and Cross-Platform Dependence."""
    print("\n" + "=" * 70)
    print("RQ2: Tail Risk and Cross-Platform Dependence")
    print("=" * 70)
    _run_module("Copula Dependence", "src.analysis.copula_dependence", dry_run=dry_run)
    _run_module("GMM/SDF Estimation", "src.analysis.gmm_sdf", ["--markets", "20"], dry_run=dry_run)


def check_artifacts():
    """Check which artifacts exist from previous runs."""
    print("\n" + "=" * 70)
    print("Artifact Status")
    print("=" * 70)
    expected = {
        "RQ1": ["phase2_results.json", "phase3_results.json", "pin_results.json"],
        "RQ2": ["copula_results.json", "gmm_sdf_results.json"],
        "RQ3": ["strategy_results.json"],
        "Docs": ["research_design_v2.md", "phase3_analysis.md"],
    }
    for cat, files in expected.items():
        print(f"\n  {cat}:")
        for f in files:
            path = ARTIFACTS / f
            if path.exists():
                print(f"    [OK]      {f} ({path.stat().st_size:,} bytes)")
            else:
                print(f"    [MISSING] {f}")


def main():
    parser = argparse.ArgumentParser(description="Run all analyses")
    parser.add_argument("--rq1", action="store_true")
    parser.add_argument("--rq2", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_artifacts()
        return

    run_all = args.all or (not args.rq1 and not args.rq2)
    if args.rq1 or run_all:
        run_rq1(dry_run=args.dry_run)
    if args.rq2 or run_all:
        run_rq2(dry_run=args.dry_run)
    if run_all:
        print("\n" + "=" * 70)
        print("RQ3: Autoresearch — run separately: make autoresearch")
        print("=" * 70)

    check_artifacts()


if __name__ == "__main__":
    main()
