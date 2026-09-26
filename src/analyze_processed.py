
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = Path("data/processed")

files = sorted(PROCESSED_DIR.glob("*.csv"))

print("=" * 70)
print("PROCESSED CIC-IDS2017 DATASET ANALYSIS")
print("=" * 70)

print(f"\nProcessed files found: {len(files)}")


# ============================================================
# GLOBAL COUNTERS
# ============================================================

target_counts = {
    0: 0,
    1: 0
}

total_rows = 0

feature_count = None


# ============================================================
# PROCESS FILES
# ============================================================

for file in files:

    print("\n" + "-" * 70)
    print(f"Analyzing: {file.name}")
    print("-" * 70)

    file_rows = 0
    benign = 0
    attack = 0

    for chunk in pd.read_csv(
        file,
        chunksize=50000
    ):

        # Number of features
        if feature_count is None:
            feature_count = len(chunk.columns) - 1

            print(f"Number of ML features: {feature_count}")

            print("\nColumns:")
            for column in chunk.columns:
                print(f"- {column}")

        # Count target
        counts = chunk["Target"].value_counts()

        benign_count = int(counts.get(0, 0))
        attack_count = int(counts.get(1, 0))

        benign += benign_count
        attack += attack_count

        target_counts[0] += benign_count
        target_counts[1] += attack_count

        rows = len(chunk)
        file_rows += rows
        total_rows += rows

    print(f"\nRows: {file_rows:,}")
    print(f"BENIGN: {benign:,}")
    print(f"ATTACK: {attack:,}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal rows: {total_rows:,}")

print(f"\nBENIGN: {target_counts[0]:,}")
print(f"ATTACK: {target_counts[1]:,}")

if total_rows > 0:

    benign_percentage = (
        target_counts[0] / total_rows
    ) * 100

    attack_percentage = (
        target_counts[1] / total_rows
    ) * 100

    print(f"\nBENIGN percentage: {benign_percentage:.2f}%")
    print(f"ATTACK percentage: {attack_percentage:.2f}%")

print(f"\nNumber of ML features: {feature_count}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)