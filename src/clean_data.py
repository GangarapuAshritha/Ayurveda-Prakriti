"""
clean_data.py
=============
Objective 1 - Step 2: Clean the integrated dataset.

Cleaning steps performed:
  1. Remove duplicate rows
  2. Handle missing values (mode imputation for categorical features)
  3. Strip whitespace from categorical values
  4. Standardize categorical values to lowercase for consistency
  5. Save the final cleaned dataset
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEGRATED_DIR = os.path.join(BASE_DIR, "data", "integrated")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_CSV = os.path.join(INTEGRATED_DIR, "integrated_dataset.csv")
OUTPUT_CSV = os.path.join(CLEANED_DIR, "cleaned_dataset.csv")
REPORT_PATH = os.path.join(RESULTS_DIR, "cleaning_report.json")


def _json_serializer(obj):
    """Handle numpy/pandas types that are not JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def get_categorical_columns(df):
    """Return categorical/text columns (object or string dtype)."""
    return df.select_dtypes(include=["object", "string"]).columns.tolist()


def strip_whitespace(df):
    """Strip leading/trailing whitespace from all categorical values."""
    for col in get_categorical_columns(df):
        df[col] = df[col].astype(str).str.strip()
    return df


def standardize_categories(df):
    """Convert categorical values to lowercase for consistency."""
    for col in get_categorical_columns(df):
        if col != "patient_id":
            df[col] = df[col].astype(str).str.lower()
    return df


def remove_duplicates(df):
    """Remove duplicate rows excluding patient_id and return cleaned dataframe."""
    before = len(df)
    subset_cols = [c for c in df.columns if c != "patient_id"]
    df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    removed = before - len(df)
    return df, removed


def impute_missing(df):
    """Impute missing values in categorical columns using the mode."""
    missing_before = int(df.isnull().sum().sum())
    imputation_log = {}

    for col in df.columns:
        n_miss = int(df[col].isnull().sum())
        if n_miss == 0:
            continue

        if df[col].dtype in ("int64", "float64"):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            imputation_log[col] = {"method": "median", "value": float(fill_val), "filled": n_miss}
        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "unknown"
            df[col] = df[col].fillna(fill_val)
            imputation_log[col] = {"method": "mode", "value": str(fill_val), "filled": n_miss}

    missing_after = int(df.isnull().sum().sum())
    return df, missing_before, missing_after, imputation_log


def clean_dataset():
    """Run the full cleaning pipeline."""
    print("=" * 60)
    print("OBJECTIVE 1 - STEP 2: DATA CLEANING")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Integrated dataset not found: {INPUT_CSV}")

    print(f"\n[1] Loading integrated dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    original_rows = len(df)
    original_cols = len(df.columns)
    print(f"    Loaded shape: {df.shape}")
    print(f"    Missing values: {df.isnull().sum().sum()}")
    print(f"    Duplicate rows: {df.duplicated().sum()}")

    print("\n[2] Stripping whitespace...")
    df = strip_whitespace(df)

    print("[3] Standardizing categorical values...")
    df = standardize_categories(df)

    print("[4] Removing duplicate rows...")
    df, dupes_removed = remove_duplicates(df)
    print(f"    Duplicates removed: {dupes_removed}")

    print("[5] Handling missing values...")
    df, missing_before, missing_after, imputation_log = impute_missing(df)
    print(f"    Missing values before: {missing_before}")
    print(f"    Missing values after: {missing_after}")

    print("[6] Final duplicate check...")
    dupes_after = int(df.duplicated().sum())
    print(f"    Duplicate rows after cleaning: {dupes_after}")

    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    report = {
        "generated_at": datetime.now().isoformat(),
        "input_file": os.path.basename(INPUT_CSV),
        "original_row_count": original_rows,
        "original_column_count": original_cols,
        "missing_values_before": missing_before,
        "duplicates_before": dupes_removed + dupes_after,
        "missing_values_after": missing_after,
        "duplicates_after": dupes_after,
        "rows_removed": dupes_removed,
        "final_row_count": len(df),
        "final_column_count": len(df.columns),
        "imputation_details": imputation_log,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=_json_serializer)

    print(f"\n[7] Cleaned dataset saved: {OUTPUT_CSV}")
    print(f"    Final shape: {df.shape}")
    print(f"    Cleaning report saved: {REPORT_PATH}")
    print("\nCleaning completed successfully.")
    return df


if __name__ == "__main__":
    clean_dataset()
