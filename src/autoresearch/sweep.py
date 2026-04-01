"""
Systematic parameter sweep for high-cost regime adaptation.
Tests 20+ configurations, logs results, identifies best.
"""
import subprocess
import re
import sys
import time

TRAIN_PY = "train.py"

# Read current train.py as template
with open(TRAIN_PY) as f:
    TEMPLATE = f.read()

CONFIGS = [
    # (name, replacements_dict)
    # Baseline (current, for reference)
    ("baseline_v2", {}),
    
    # Reduce position size via SIGNAL_SCALE
    ("scale_0.1", {"SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.10"}),
    ("scale_0.3", {"SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.30"}),
    ("scale_0.5", {"SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.50"}),
    ("scale_0.05", {"SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.05"}),
    
    # Increase turnover threshold
    ("turnover_0.90", {"TURNOVER_THRESHOLD = 0.75": "TURNOVER_THRESHOLD = 0.90"}),
    ("turnover_0.95", {"TURNOVER_THRESHOLD = 0.75": "TURNOVER_THRESHOLD = 0.95"}),
    ("turnover_0.99", {"TURNOVER_THRESHOLD = 0.75": "TURNOVER_THRESHOLD = 0.99"}),
    
    # Combine: small positions + high threshold
    ("small_pos_high_thresh", {
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.30",
        "TURNOVER_THRESHOLD = 0.75": "TURNOVER_THRESHOLD = 0.20",
    }),
    ("tiny_pos", {
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.05",
        "TURNOVER_THRESHOLD = 0.75": "TURNOVER_THRESHOLD = 0.04",
    }),
    
    # No direction model (reduces noise)
    ("no_direction", {"DIR_BLEND = 0.45": "DIR_BLEND = 0.00"}),
    ("no_dir_small_pos", {
        "DIR_BLEND = 0.45": "DIR_BLEND = 0.00",
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.30",
    }),
    
    # No tail scaling
    ("no_tail", {"TAIL_SCALE_FACTOR = 0.3": "TAIL_SCALE_FACTOR = 0.0"}),
    
    # Maximum turnover suppression with tiny positions
    ("ultra_conservative", {
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.02",
        "TURNOVER_THRESHOLD = 0.75": "TURNOVER_THRESHOLD = 0.01",
        "DIR_BLEND = 0.45": "DIR_BLEND = 0.00",
        "TAIL_SCALE_FACTOR = 0.3": "TAIL_SCALE_FACTOR = 0.0",
    }),
    
    # Enter once, hold (threshold allows first entry from 0 but blocks changes)
    ("enter_hold_v1", {
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 5.00",
        "TURNOVER_THRESHOLD = 0.75": "TURNOVER_THRESHOLD = 0.99",
        "DIR_BLEND = 0.45": "DIR_BLEND = 0.00",
        "TAIL_SCALE_FACTOR = 0.3": "TAIL_SCALE_FACTOR = 0.0",
    }),
    ("enter_hold_v2", {
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 10.00",
        "TURNOVER_THRESHOLD = 0.75": "TURNOVER_THRESHOLD = 0.99",
        "DIR_BLEND = 0.45": "DIR_BLEND = 0.00",
        "TAIL_SCALE_FACTOR = 0.3": "TAIL_SCALE_FACTOR = 0.0",
    }),
    
    # Moderate positions, high threshold
    ("moderate_hold", {
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 1.00",
        "TURNOVER_THRESHOLD = 0.75": "TURNOVER_THRESHOLD = 0.90",
    }),
    
    # Aggressive boundary play (prices near 0/1 have stronger reversion)
    ("boundary_heavy", {
        "BOUNDARY_COEFF = 2.0": "BOUNDARY_COEFF = 5.0",
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.20",
    }),
    
    # Pure ML direction with tiny positions
    ("pure_ml_tiny", {
        "DIR_BLEND = 0.45": "DIR_BLEND = 1.00",
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.10",
    }),
    
    # Momentum instead of mean reversion
    ("momentum_v1", {
        "SIGNAL_SCALE = 1.50": "SIGNAL_SCALE = 0.10",
    }),
]

# Momentum config needs a code change too — handle separately
MOMENTUM_CODE_CHANGE = (
    "-np.sign(combined_ret) * np.sqrt(np.abs(combined_ret))",
    "np.sign(combined_ret) * np.sqrt(np.abs(combined_ret))"
)

print(f"Running {len(CONFIGS)} experiments...")
print(f"{'#':>3} {'Name':30} {'Sharpe':>10} {'MeanPos':>10} {'WinRate':>10} {'TotalRet':>12}")
print("-" * 85)

results = []
for i, (name, replacements) in enumerate(CONFIGS):
    # Construct modified train.py
    code = TEMPLATE
    for old, new in replacements.items():
        code = code.replace(old, new)
    
    # Special handling for momentum experiment
    if name == "momentum_v1":
        code = code.replace(MOMENTUM_CODE_CHANGE[0], MOMENTUM_CODE_CHANGE[1])
    
    with open(TRAIN_PY, "w") as f:
        f.write(code)
    
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_PY],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        
        sharpe = float(re.search(r"sharpe_ratio:\s+([-\d.]+)", output).group(1))
        mean_pos = float(re.search(r"mean_position:\s+([\d.]+)", output).group(1))
        win_rate = float(re.search(r"win_rate:\s+([\d.]+)", output).group(1))
        total_ret = float(re.search(r"total_return:\s+([-\d.]+)", output).group(1))
        
        print(f"{i+1:>3} {name:30} {sharpe:>10.4f} {mean_pos:>10.4f} {win_rate:>10.4f} {total_ret:>12.2f}")
        results.append((name, sharpe, mean_pos, win_rate, total_ret))
    except Exception as e:
        print(f"{i+1:>3} {name:30} ERROR: {e}")
        results.append((name, None, None, None, None))

# Restore original train.py
with open(TRAIN_PY, "w") as f:
    f.write(TEMPLATE)

# Sort by Sharpe (best first)
valid = [(n, s, mp, wr, tr) for n, s, mp, wr, tr in results if s is not None]
valid.sort(key=lambda x: -x[1])

print("\n" + "=" * 85)
print("RANKED RESULTS (best Sharpe first):")
print(f"{'Rank':>4} {'Name':30} {'Sharpe':>10} {'MeanPos':>10} {'TotalRet':>12}")
for rank, (n, s, mp, wr, tr) in enumerate(valid, 1):
    print(f"{rank:>4} {n:30} {s:>10.4f} {mp:>10.4f} {tr:>12.2f}")
