
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/training")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50000

# Number of samples from each class
SAMPLES_PER_CLASS = 250000


# ============================================================
# FIND FILES
# ============================================================

files = sorted(PROCESSED_DIR.glob("*.csv"))

print("=" * 70)
print("PREPARING ML TRAINING DATA")
print("=" * 70)

print(f"\nProcessed files found: {len(files)}")


# ============================================================
# STORAGE
# ============================================================

benign_parts = []
attack_parts = []

benign_count = 0
attack_count = 0


# ============================================================
# READ DATA
# ============================================================

for file in files:

    print(f"\nReading: {file.name}")

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE
    ):

        # ----------------------------------------------------
        # BENIGN
        # ----------------------------------------------------

        if benign_count < SAMPLES_PER_CLASS:

            benign_rows = chunk[
                chunk["Target"] == 0
            ]

            remaining = (
                SAMPLES_PER_CLASS - benign_count
            )

            benign_rows = benign_rows.iloc[
                :remaining
            ]

            if len(benign_rows) > 0:

                benign_parts.append(
                    benign_rows
                )

                benign_count += len(
                    benign_rows
                )


        # ----------------------------------------------------
        # ATTACK
        # ----------------------------------------------------

        if attack_count < SAMPLES_PER_CLASS:

            attack_rows = chunk[
                chunk["Target"] == 1
            ]

            remaining = (
                SAMPLES_PER_CLASS - attack_count
            )

            attack_rows = attack_rows.iloc[
                :remaining
            ]

            if len(attack_rows) > 0:

                attack_parts.append(
                    attack_rows
                )

                attack_count += len(
                    attack_rows
                )


        # ----------------------------------------------------
        # Stop when both classes are ready
        # ----------------------------------------------------

        if (
            benign_count >= SAMPLES_PER_CLASS
            and
            attack_count >= SAMPLES_PER_CLASS
        ):
            break

    print(
        f"BENIGN collected: {benign_count:,}"
    )

    print(
        f"ATTACK collected: {attack_count:,}"
    )

    if (
        benign_count >= SAMPLES_PER_CLASS
        and
        attack_count >= SAMPLES_PER_CLASS
    ):
        break


# ============================================================
# COMBINE
# ============================================================

print("\n" + "=" * 70)
print("COMBINING DATA")
print("=" * 70)

benign_df = pd.concat(
    benign_parts,
    ignore_index=True
)

attack_df = pd.concat(
    attack_parts,
    ignore_index=True
)


# ============================================================
# FINAL DATASET
# ============================================================

df = pd.concat(
    [
        benign_df,
        attack_df
    ],
    ignore_index=True
)


# Shuffle the dataset

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR /
    "nids_training_dataset.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING DATASET CREATED")
print("=" * 70)

print(f"\nTotal rows: {len(df):,}")

print("\nClass distribution:")

print(
    df["Target"].value_counts()
)

print("\nPercentage:")

print(
    df["Target"]
    .value_counts(normalize=True)
    .mul(100)
)

print(f"\nSaved to:")
print(output_file)

print("\nDataset preparation completed.")