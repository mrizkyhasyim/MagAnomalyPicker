import sys
from pathlib import Path

# Add project root (MagAnomalyPicker) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from core.io.loader import load_dataset
from core.nav.cleaner import process_altitude, process_navigation


def run_phase1_benchmark():
    raw_path = PROJECT_ROOT / "data" / "raw_data.csv"
    benchmark_path = PROJECT_ROOT / "data" / "montaj_benchmark.csv"

    print("--- Phase 1: Navigation & Altimeter Engine Benchmark ---")

    # 1. Load Datasets
    df_raw = load_dataset(raw_path)
    df_benchmark = load_dataset(benchmark_path)

    line_col = "Line" if "Line" in df_raw.columns else None

    # 2. Run Engine Pipeline
    df_proc = process_navigation(df_raw, line_col=line_col)
    df_proc = process_altitude(df_proc, line_col=line_col)

    # 3. Channels to Compare
    target_channels = [
        "Easting_DS",
        "Easting_LP",
        "Northing_DS",
        "Northing_LP",
        "DistQC",
        "Alt_DS",
        "Alt_RS",
    ]

    print(f"\n{'Channel':<15} | {'Max Delta':<12} | {'MAE':<12} | {'Status'}")
    print("-" * 55)

    all_passed = True
    tolerance = 1e-3  # 1 mm tolerance limit

    for col in target_channels:
        if col not in df_benchmark.columns:
            print(f"{col:<15} | {'N/A (Missing in Benchmark)':<27} | ⚠️ SKIP")
            continue

        # Delta Calculation: Python Result - Montaj Reference
        delta = (df_proc[col] - df_benchmark[col]).abs()
        max_delta = delta.max(skipna=True)
        mae = delta.mean(skipna=True)

        status = "✅ PASS" if max_delta <= tolerance else "❌ FAIL"
        if max_delta > tolerance:
            all_passed = False

        print(f"{col:<15} | {max_delta:<12.6f} | {mae:<12.6f} | {status}")

    print("-" * 55)
    if all_passed:
        print("\n🎉 PHASE 1 VERIFICATION SUCCESSFUL: All channels match Montaj!")
    else:
        print("\n⚠️ DISCREPANCY DETECTED: Review filter widths or edge padding.")


if __name__ == "__main__":
    run_phase1_benchmark()