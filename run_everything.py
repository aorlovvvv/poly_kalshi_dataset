#!/usr/bin/env python3
"""
Single-command runner for all analyses in the prediction market research project.

Usage:
    python run_everything.py              # Run all analyses
    python run_everything.py --sharpe     # Just validate Sharpe
    python run_everything.py --pin        # Just run PIN estimation
    python run_everything.py --copula     # Just run copula analysis
    python run_everything.py --gmm        # Just run GMM/SDF
    python run_everything.py --ppo        # Just test PPO vs LogReg
    python run_everything.py --skip-sharpe  # All except Sharpe (if already done)

Requires: duckdb, numpy, pandas, scipy, scikit-learn
Optional: torch (for PPO), xgboost (for phase2 ML)

All output goes to artifacts/ directory.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PYTHON = sys.executable


def banner(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def run_cmd(desc: str, cmd: list, cwd: str = None, timeout: int = 600) -> bool:
    """Run a command with timing and error reporting."""
    print(f">>> {desc}")
    print(f"    Command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=cwd or str(ROOT),
            timeout=timeout,
            capture_output=False  # Let output flow to terminal
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"    OK ({elapsed:.1f}s)")
            return True
        else:
            print(f"    FAILED (exit {result.returncode}, {elapsed:.1f}s)")
            return False
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def step_validate_sharpe():
    """Step 1: Rebuild cache and validate Sharpe with fixed harness."""
    banner("STEP 1: Validate Sharpe (50 markets, 7c costs, market resets)")

    cache_dir = os.path.expanduser("~/.cache/autoresearch_pm")
    cache_file = os.path.join(cache_dir, "kalshi_features.parquet")

    # Delete old cache to force rebuild with MAX_MARKETS=50
    if os.path.exists(cache_file):
        print(f"Deleting old cache: {cache_file}")
        os.remove(cache_file)

    # Rebuild cache
    ok = run_cmd(
        "Rebuild feature cache (50 markets)",
        [PYTHON, "-m", "src.autoresearch.prepare", "--force"],
        timeout=300
    )
    if not ok:
        print("Cache rebuild failed!")
        return False

    # Run train.py with LogReg baseline
    ok = run_cmd(
        "Run LogReg strategy evaluation",
        [PYTHON, "-m", "src.autoresearch.train"],
        timeout=300
    )

    return ok


def step_test_ppo():
    """Step 2: Test PPO vs LogReg."""
    banner("STEP 2: Test PPO Strategy (if torch available)")

    # Check torch availability
    try:
        import torch
        has_torch = True
        print(f"PyTorch available: {torch.__version__}")
    except ImportError:
        has_torch = False
        print("PyTorch not available — skipping PPO test")
        return True  # Not a failure, just skipped

    if has_torch:
        ok = run_cmd(
            "Run PPO strategy evaluation",
            [PYTHON, "-m", "src.autoresearch.train", "--strategy", "ppo"],
            timeout=600
        )
        return ok

    return True


def step_pin_estimation():
    """Step 3: Run PIN estimation on real Kalshi data."""
    banner("STEP 3: PIN/VPIN Estimation (30 markets)")

    ok = run_cmd(
        "PIN estimation (Easley-O'Hara 1996)",
        [PYTHON, "-m", "src.analysis.pin_estimation", "--markets", "30"],
        timeout=600
    )

    if ok:
        pin_path = ARTIFACTS / "pin_results.json"
        if pin_path.exists():
            print(f"\n    Output: {pin_path} ({pin_path.stat().st_size:,} bytes)")

    return ok


def step_copula_analysis():
    """Step 4: Run copula analysis on matched cross-platform events."""
    banner("STEP 4: Copula Cross-Platform Dependence")

    ok = run_cmd(
        "Copula analysis (matched events)",
        [PYTHON, "-m", "src.analysis.copula_dependence"],
        timeout=600
    )

    if ok:
        cop_path = ARTIFACTS / "copula_results.json"
        if cop_path.exists():
            print(f"\n    Output: {cop_path} ({cop_path.stat().st_size:,} bytes)")

    return ok


def step_gmm_sdf():
    """Step 5: Run GMM/SDF estimation."""
    banner("STEP 5: GMM/SDF Estimation")

    ok = run_cmd(
        "GMM/SDF (Hansen 1982)",
        [PYTHON, "-m", "src.analysis.gmm_sdf", "--markets", "30"],
        timeout=300
    )

    if ok:
        gmm_path = ARTIFACTS / "gmm_sdf_results.json"
        if gmm_path.exists():
            print(f"\n    Output: {gmm_path} ({gmm_path.stat().st_size:,} bytes)")

    return ok


def check_artifacts():
    """Check all expected output artifacts."""
    banner("ARTIFACT CHECK")

    expected = {
        "RQ1 (Efficiency + Informed Trading)": [
            "phase2_results.json",
            "phase3_results.json",
            "pin_results.json",
        ],
        "RQ2 (Tails + Cross-Platform)": [
            "copula_results.json",
            "gmm_sdf_results.json",
        ],
        "RQ3 (Autoresearch)": [
            "strategy_results.json",
        ],
    }

    all_ok = True
    for rq, files in expected.items():
        print(f"\n  {rq}:")
        for f in files:
            path = ARTIFACTS / f
            if path.exists():
                print(f"    [OK]      {f} ({path.stat().st_size:,} bytes)")
            else:
                print(f"    [MISSING] {f}")
                all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Run all prediction market analyses")
    parser.add_argument("--sharpe", action="store_true", help="Only: validate Sharpe")
    parser.add_argument("--ppo", action="store_true", help="Only: test PPO")
    parser.add_argument("--pin", action="store_true", help="Only: PIN estimation")
    parser.add_argument("--copula", action="store_true", help="Only: copula analysis")
    parser.add_argument("--gmm", action="store_true", help="Only: GMM/SDF")
    parser.add_argument("--skip-sharpe", action="store_true", help="Skip Sharpe validation")
    parser.add_argument("--check", action="store_true", help="Only check artifacts")
    args = parser.parse_args()

    os.makedirs(ARTIFACTS, exist_ok=True)

    if args.check:
        check_artifacts()
        return

    # Selective mode
    selective = any([args.sharpe, args.ppo, args.pin, args.copula, args.gmm])

    results = {}
    t_start = time.time()

    if args.sharpe or (not selective and not args.skip_sharpe):
        results["sharpe"] = step_validate_sharpe()

    if args.ppo or not selective:
        results["ppo"] = step_test_ppo()

    if args.pin or not selective:
        results["pin"] = step_pin_estimation()

    if args.copula or not selective:
        results["copula"] = step_copula_analysis()

    if args.gmm or not selective:
        results["gmm"] = step_gmm_sdf()

    elapsed = time.time() - t_start

    # Summary
    banner(f"SUMMARY (total: {elapsed:.0f}s)")
    for step, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {step}")

    print()
    check_artifacts()

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  FAILURES: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n  All steps passed!")


if __name__ == "__main__":
    main()
