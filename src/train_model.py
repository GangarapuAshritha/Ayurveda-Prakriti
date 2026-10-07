"""
train_model.py
==============
Objective 2 - Step 1: Train an AI/ML classification model on the cleaned dataset.

Target column: dosha (Vata / Pitta / Kapha)
Model: Random Forest Classifier with One-Hot Encoding for categorical features.
Output: Saved trained model in models/trained_model.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")

INPUT_CSV = os.path.join(CLEANED_DIR, "cleaned_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "trained_model.pkl")

TARGET_COL = "dosha"
EXCLUDE_COLS = {"patient_id", "dosha"}
RANDOM_STATE = 42
TEST_SIZE = 0.20


def identify_column_types(df):
    """Identify numeric and categorical feature columns."""
    feature_df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "string"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols):
    """Build a scikit-learn pipeline with preprocessing and Random Forest."""
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    return pipeline


def train_model():
    """Train and save the classification model."""
    print("=" * 60)
    print("OBJECTIVE 2 - STEP 1: MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT_CSV}")

    print(f"\n[1] Loading cleaned dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"    Loaded shape: {df.shape}")

    print("\n[2] Identifying target and features...")
    target_col_actual = None
    for col in df.columns:
        if col.lower() == TARGET_COL.lower():
            target_col_actual = col
            break

    if target_col_actual is None:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    print(f"    Target column: {target_col_actual}")
    y = df[target_col_actual]
    class_names = sorted(y.unique().tolist())
    print(f"    Classes ({len(class_names)}): {class_names}")

    numeric_cols, categorical_cols = identify_column_types(df)
    print(f"    Numeric features:     {len(numeric_cols)}")
    print(f"    Categorical features: {len(categorical_cols)}")

    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()

    print("\n[3] Splitting data (80/20 with stratification)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"    Train samples: {len(X_train)}")
    print(f"    Test samples:  {len(X_test)}")

    print("\n[4] Training Random Forest Classifier...")
    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\n[5] Model saved: {MODEL_PATH}")
    print("\nTraining completed successfully.")
    return pipeline, X_test, y_test, class_names


if __name__ == "__main__":
    train_model()
