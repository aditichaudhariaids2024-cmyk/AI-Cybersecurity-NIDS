
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/TrafficLabelling")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50000

# Columns that identify a network flow rather than describe
# its numerical traffic behavior.
DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]


# ============================================================
# FIND DATASET FILES
# ============================================================

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("CIC-IDS2017 PREPROCESSING")
print("=" * 70)

print(f"\nFiles found: {len(csv_files)}")


# ============================================================
# PROCESS EACH FILE
# ============================================================

processed_files = []

for file in csv_files:

    print("\n" + "-" * 70)
    print(f"Processing: {file.name}")
    print("-" * 70)

    output_file = OUTPUT_DIR / f"processed_{file.name}"

    first_chunk = True
    total_rows = 0
    removed_rows = 0

    try:

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="latin1"
        ):

            original_rows = len(chunk)

            # ------------------------------------------------
            # 1. Clean column names
            # ------------------------------------------------

            chunk.columns = chunk.columns.str.strip()


            # ------------------------------------------------
            # 2. Clean Label
            # ------------------------------------------------

            chunk["Label"] = (
                chunk["Label"]
                .astype(str)
                .str.strip()
            )


            # ------------------------------------------------
            # 3. Convert attack labels into binary labels
            # ------------------------------------------------

            chunk["Target"] = np.where(
                chunk["Label"] == "BENIGN",
                0,
                1
            )


            # ------------------------------------------------
            # 4. Remove identifier columns
            # ------------------------------------------------

            columns_to_drop = [
                col for col in DROP_COLUMNS
                if col in chunk.columns
            ]

            chunk = chunk.drop(
                columns=columns_to_drop
            )


            # ------------------------------------------------
            # 5. Remove original Label
            # ------------------------------------------------

            chunk = chunk.drop(
                columns=["Label"]
            )


            # ------------------------------------------------
            # 6. Replace infinity values
            # ------------------------------------------------

            chunk = chunk.replace(
                [np.inf, -np.inf],
                np.nan
            )


            # ------------------------------------------------
            # 7. Convert feature columns to numeric
            # ------------------------------------------------

            feature_columns = [
                col for col in chunk.columns
                if col != "Target"
            ]

            for col in feature_columns:

                chunk[col] = pd.to_numeric(
                    chunk[col],
                    errors="coerce"
                )


            # ------------------------------------------------
            # 8. Remove rows containing invalid values
            # ------------------------------------------------

            before_drop = len(chunk)

            chunk = chunk.dropna()

            removed_rows += (
                before_drop - len(chunk)
            )


            # ------------------------------------------------
            # 9. Remove duplicate rows
            # ------------------------------------------------

            chunk = chunk.drop_duplicates()


            # ------------------------------------------------
            # 10. Save processed chunk
            # ------------------------------------------------

            chunk.to_csv(
                output_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False

            total_rows += len(chunk)

            print(
                f"Processed rows: {total_rows:,}",
                end="\r"
            )


        processed_files.append(output_file)

        print()
        print(f"Saved: {output_file}")
        print(f"Valid rows: {total_rows:,}")
        print(f"Invalid rows removed: {removed_rows:,}")


    except Exception as e:

        print()
        print(f"ERROR: {e}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print(f"\nProcessed files: {len(processed_files)}")

for file in processed_files:
    print(f"- {file}")