import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)


# ============================================================
# PATHS
# ============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

import pandas as pd

features_df = pd.read_csv(DATA_DIR / "features.csv")

feature_names = [
    "Mean",
    "Std",
    "Variance",
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma"
]

features = features_df[feature_names]
labels = features_df["Label"]

print("Features Shape:", features.shape)
print("Labels Shape:", labels.shape)

print("=" * 60)
print("EEG SEIZURE DETECTION - FINAL EVALUATION")
print("=" * 60)

print("\nFeatures Shape:", features.shape)
print("Labels Shape:", labels.shape)


# ============================================================
# 2. LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("\nModel loaded successfully!")


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Class Distribution:")
print(np.unique(y_train, return_counts=True))

print("\nTesting Class Distribution:")
print(np.unique(y_test, return_counts=True))


# ============================================================
# 4. DEFAULT MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test)

print("\nPredicted Class Distribution:")
print(np.unique(y_pred, return_counts=True))


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Seizure"],
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# 7. METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

sensitivity = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 8. SPECIFICITY
# ============================================================

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)


# ============================================================
# 9. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL METRICS")
print("=" * 60)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Sensitivity       : {sensitivity:.4f}")
print(f"Specificity       : {specificity:.4f}")
print(f"F1-Score          : {f1:.4f}")
print(f"Balanced Accuracy : {balanced_accuracy:.4f}")


# ============================================================
# 10. SAVE FINAL EVALUATION
# ============================================================

evaluation_file = os.path.join(
    RESULTS_DIR,
    "final_evaluation.txt"
)

with open(evaluation_file, "w") as f:

    f.write("EEG SEIZURE DETECTION - FINAL EVALUATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Features Shape: {features.shape}\n")
    f.write(f"Labels Shape: {labels.shape}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\nFinal Metrics:\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Sensitivity: {sensitivity:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")

print("\nSaved:")
print(evaluation_file)


# ============================================================
# 11. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Normal", "Seizure"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Seizure"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

confusion_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300
)

plt.close()

print("Saved:", confusion_path)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

feature_names = [
    "Mean",
    "Std",
    "Variance",
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma"
]

if hasattr(model, "feature_importances_"):

    importances = model.feature_importances_

    sorted_indices = np.argsort(importances)

    plt.figure(figsize=(8, 5))

    plt.barh(
        np.array(feature_names)[sorted_indices],
        importances[sorted_indices]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    feature_path = os.path.join(
        RESULTS_DIR,
        "feature_importance.png"
    )

    plt.savefig(
        feature_path,
        dpi=300
    )

    plt.close()

    print("Saved:", feature_path)


# ============================================================
# 13. THRESHOLD ANALYSIS
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]

thresholds = [
    0.50,
    0.20,
    0.10,
    0.05,
    0.01
]

sensitivities = []
specificities = []

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

for threshold in thresholds:

    threshold_predictions = (
        probabilities >= threshold
    ).astype(int)

    threshold_cm = confusion_matrix(
        y_test,
        threshold_predictions
    )

    tn_t, fp_t, fn_t, tp_t = threshold_cm.ravel()

    sensitivity_t = (
        tp_t / (tp_t + fn_t)
        if (tp_t + fn_t) > 0
        else 0
    )

    specificity_t = (
        tn_t / (tn_t + fp_t)
        if (tn_t + fp_t) > 0
        else 0
    )

    sensitivities.append(sensitivity_t)
    specificities.append(specificity_t)

    print(
        f"\nThreshold: {threshold}"
    )

    print(
        "Confusion Matrix:"
    )

    print(
        threshold_cm
    )

    print(
        f"Sensitivity: {sensitivity_t:.4f}"
    )

    print(
        f"Specificity: {specificity_t:.4f}"
    )


# ============================================================
# 14. THRESHOLD PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    sensitivities,
    marker="o",
    label="Sensitivity"
)

plt.plot(
    thresholds,
    specificities,
    marker="o",
    label="Specificity"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.title(
    "Sensitivity and Specificity vs Classification Threshold"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

threshold_path = os.path.join(
    RESULTS_DIR,
    "threshold_analysis.png"
)

plt.savefig(
    threshold_path,
    dpi=300
)

plt.close()

print("\nSaved:", threshold_path)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)