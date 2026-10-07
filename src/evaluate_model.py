"""
evaluate_model.py
=================
Objective 2 - Step 2: Evaluate the trained model.

Metrics computed:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - Stratified K-Fold Cross-Validation

Outputs saved in results/:
  - accuracy_results.txt
  - classification_report.txt
  - confusion_matrix.png
  - evaluation_report.json
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_CSV = os.path.join(CLEANED_DIR, "cleaned_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "trained_model.pkl")

TARGET_COL = "dosha"
EXCLUDE_COLS = {"patient_id", "dosha"}
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def _json_serializer(obj):
    """Handle numpy/pandas types that are not JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Confusion matrix plot saved: {output_path}")


def evaluate_model():
    """Load trained model and evaluate on test data with cross-validation."""
    print("=" * 60)
    print("OBJECTIVE 2 - STEP 2: MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT_CSV}")

    print(f"\n[1] Loading cleaned dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    print(f"[2] Loading trained model: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    # Identify target
    target_col_actual = None
    for col in df.columns:
        if col.lower() == TARGET_COL.lower():
            target_col_actual = col
            break

    if target_col_actual is None:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    y = df[target_col_actual]
    class_names = sorted(y.unique().tolist())

    feature_df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    X = feature_df.copy()

    print("\n[3] Splitting data for evaluation (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n[4] Making predictions on test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-score:  {f1:.4f}")

    print("\n[5] Stratified K-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(f"    CV fold scores: {[round(float(s), 4) for s in cv_scores]}")
    print(f"    CV mean accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save accuracy results
    accuracy_path = os.path.join(RESULTS_DIR, "accuracy_results.txt")
    with open(accuracy_path, "w") as f:
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Target column: {target_col_actual}\n")
        f.write(f"Classes:       {class_names}\n")
        f.write(f"Train size:    {len(X_train)}\n")
        f.write(f"Test size:     {len(X_test)}\n\n")
        f.write(f"Test Accuracy:  {acc:.4f}\n")
        f.write(f"Precision:      {prec:.4f}\n")
        f.write(f"Recall:         {rec:.4f}\n")
        f.write(f"F1-score:       {f1:.4f}\n\n")
        f.write(f"CV Mean Accuracy ({CV_FOLDS}-fold): {cv_mean:.4f} +/- {cv_std:.4f}\n")
    print(f"\n[6] Accuracy results saved: {accuracy_path}")

    # Save classification report
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"    Classification report saved: {report_path}")

    # Save confusion matrix plot
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # Save JSON evaluation report
    json_path = os.path.join(RESULTS_DIR, "evaluation_report.json")
    evaluation = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "target_column": target_col_actual,
        "class_names": class_names,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "cv_mean_accuracy": round(cv_mean, 4),
        "cv_std_accuracy": round(cv_std, 4),
        "confusion_matrix": cm.tolist(),
        "cv_fold_scores": [round(float(s), 4) for s in cv_scores],
    }
    with open(json_path, "w") as f:
        json.dump(evaluation, f, indent=2, default=_json_serializer)
    print(f"    JSON evaluation report saved: {json_path}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Test Accuracy:       {acc:.4f}")
    print(f"  Precision:           {prec:.4f}")
    print(f"  Recall:              {rec:.4f}")
    print(f"  F1-score:            {f1:.4f}")
    print(f"  CV Mean Accuracy:    {cv_mean:.4f} +/- {cv_std:.4f}")
    print("\nEvaluation completed successfully.")
    return evaluation


if __name__ == "__main__":
    evaluate_model()
