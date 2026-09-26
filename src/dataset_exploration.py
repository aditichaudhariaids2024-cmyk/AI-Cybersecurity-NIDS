from pathlib import Path
import pandas as pd


# ============================================================
# CIC-IDS2017 DATASET ANALYSIS
# ============================================================

DATA_DIR = Path("data/TrafficLabelling")

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("CIC-IDS2017 DATASET ANALYSIS")
print("=" * 70)

print(f"\nNumber of CSV files: {len(csv_files)}")


# ============================================================
# 1. FILE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FILE INFORMATION")
print("=" * 70)

total_size = 0

for file in csv_files:

    size_mb = file.stat().st_size / (1024 * 1024)
    total_size += size_mb

    print(f"{file.name}")
    print(f"Size: {size_mb:.2f} MB")
    print()

print(f"Approximate total dataset size: {total_size:.2f} MB")


# ============================================================
# 2. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

label_counts = {}

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    try:

        # Read the CSV in chunks
        for chunk in pd.read_csv(
            file,
            chunksize=50000,
            low_memory=False,
            encoding="latin1"
        ):

            # Remove unwanted spaces from column names
            chunk.columns = chunk.columns.str.strip()

            # Clean label values
            labels = chunk["Label"].astype(str).str.strip()

            # Count labels
            counts = labels.value_counts()

            for label, count in counts.items():

                if label not in label_counts:
                    label_counts[label] = 0

                label_counts[label] += count

    except Exception as e:

        print(f"\nERROR while processing {file.name}")
        print(e)


# ============================================================
# 3. LABEL SUMMARY
# ============================================================

label_series = pd.Series(label_counts).sort_values(
    ascending=False
)

print("\n" + "=" * 70)
print("COMPLETE LABEL DISTRIBUTION")
print("=" * 70)

print(label_series)


# ============================================================
# 4. TOTAL FLOWS
# ============================================================

total_flows = label_series.sum()

print("\n" + "=" * 70)
print("TOTAL NETWORK FLOWS")
print("=" * 70)

print(f"Total network flows: {total_flows:,}")


# ============================================================
# 5. BENIGN VS ATTACK
# ============================================================

benign_count = label_series.get("BENIGN", 0)

attack_count = total_flows - benign_count

print("\n" + "=" * 70)
print("BENIGN VS ATTACK")
print("=" * 70)

print(f"BENIGN flows : {benign_count:,}")
print(f"ATTACK flows : {attack_count:,}")


if total_flows > 0:

    benign_percentage = (
        benign_count / total_flows
    ) * 100

    attack_percentage = (
        attack_count / total_flows
    ) * 100

    print(f"\nBENIGN percentage : {benign_percentage:.2f}%")
    print(f"ATTACK percentage : {attack_percentage:.2f}%")


# ============================================================
# 6. LABELS FOUND
# ============================================================

print("\n" + "=" * 70)
print("ATTACK LABELS FOUND")
print("=" * 70)

for label in label_series.index:

    print(f"- {label}")


print("\n" + "=" * 70)
print("DATASET ANALYSIS COMPLETED")
print("=" * 70)