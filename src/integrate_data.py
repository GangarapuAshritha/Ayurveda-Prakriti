"""
integrate_data.py
=================
Objective 1 - Step 1: Integrate available raw datasets.

In this Phase 1 project, only one Ayurvedic dataset is available.
The integration step standardizes column names and prepares a single
unified dataset. If additional clinical or lifestyle datasets are added
to data/raw/ later, this script can be extended to merge them using a
common patient identifier.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INTEGRATED_DIR = os.path.join(BASE_DIR, "data", "integrated")

RAW_DATASET = os.path.join(RAW_DIR, "ayurvedic_dataset.csv")
OUTPUT_CSV = os.path.join(INTEGRATED_DIR, "integrated_dataset.csv")


def standardize_columns(df):
    """Standardize column names: lowercase, strip spaces, replace spaces with underscores."""
    df.columns = [
        col.strip().lower().replace(" ", "_") for col in df.columns
    ]
    return df


def integrate_datasets():
    """Load raw dataset(s) and produce the integrated dataset."""
    print("=" * 60)
    print("OBJECTIVE 1 - STEP 1: DATASET INTEGRATION")
    print("=" * 60)

    if not os.path.exists(RAW_DATASET):
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATASET}")

    print(f"\n[1] Loading raw dataset: {RAW_DATASET}")
    df = pd.read_csv(RAW_DATASET)
    print(f"    Loaded shape: {df.shape}")

    print("\n[2] Standardizing column names...")
    df = standardize_columns(df)
    print(f"    Columns: {list(df.columns)}")

    print("\n[3] Removing duplicate rows before integration...")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"    Duplicates removed: {removed}")

    # Add a synthetic patient_id for traceability after duplicate removal
    df.insert(0, "patient_id", [f"P{i+1:04d}" for i in range(len(df))])

    os.makedirs(INTEGRATED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[3] Integrated dataset saved: {OUTPUT_CSV}")
    print(f"    Final shape: {df.shape}")
    print("\nIntegration completed successfully.")
    return df


if __name__ == "__main__":
    integrate_datasets()
