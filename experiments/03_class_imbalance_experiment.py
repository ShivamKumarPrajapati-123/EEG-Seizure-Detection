# ============================================================
# EXPERIMENT 03: CLASS-IMBALANCE-AWARE RANDOM FOREST
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

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
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURES_PATH = DATA_DIR / "features.csv"
METADATA_PATH = DATA_DIR / "window_metadata.csv"

# Original model used in Experiment 02
ORIGINAL_MODEL_PATH = (
    MODEL_DIR / "random_forest_model.pkl"
)

# New class-weighted model
NEW_MODEL_PATH = (
    MODEL_DIR / "random_forest_class_weighted.pkl"
)

# Results
RESULTS_PATH = (
    RESULTS_DIR /
    "experiment_03_class_imbalance_evaluation.txt"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR /
    "experiment_03_class_imbalance_confusion_matrix.png"
)


# ============================================================
# 2. EXPERIMENT INFORMATION
# ============================================================

print("=" * 60)
print("EXPERIMENT 03: CLASS-IMBALANCE-AWARE MODELING")
print("=" * 60)

print("\nObjective")
print("-" * 60)

print(
    "This experiment investigates whether class imbalance handling "
    "can improve seizure detection performance."
)

print(
    "\nThe experiment uses a recording-wise train-test split to "
    "evaluate the model on unseen EEG recordings."
)

print(
    "\nA class-weighted Random Forest model is trained using "
    "class_weight='balanced'."
)

print(
    "\nThe model is evaluated using sensitivity, specificity, "
    "precision, F1-score, balanced accuracy, and confusion matrix."
)

print(
    "\nImportant: This experiment is part of a research and learning "
    "prototype and is not a clinically validated diagnostic system."
)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

print("\n" + "=" * 60)
print("CHECKING REQUIRED FILES")
print("=" * 60)

required_files = [
    FEATURES_PATH,
    METADATA_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    print("Found:", file_path.name)


print("\nAll required files found successfully.")


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

features_df = pd.read_csv(
    FEATURES_PATH
)

metadata_df = pd.read_csv(
    METADATA_PATH
)

print(
    "\nFeature Dataset Shape:",
    features_df.shape
)

print(
    "Metadata Shape:",
    metadata_df.shape
)


# ============================================================
# 5. CHECK DATA CONSISTENCY
# ============================================================

if len(features_df) != len(metadata_df):

    raise ValueError(
        "Feature dataset and metadata have different "
        "numbers of rows."
    )


print(
    "\nDataset consistency check passed."
)


# ============================================================
# 6. DEFINE FEATURES AND LABELS
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

X = features_df[
    feature_names
]

y = features_df[
    "Label"
]


print(
    "\nFeatures Shape:",
    X.shape
)

print(
    "Labels Shape:",
    y.shape
)


# ============================================================
# 7. RECORDING-WISE SPLIT
# ============================================================

print("\n" + "=" * 60)
print("RECORDING-WISE TRAIN / TEST SPLIT")
print("=" * 60)


# Same recording split used in Experiment 02
# This ensures fair comparison between experiments.

train_recordings = [
    "chb01_03.edf",
    "chb01_04.edf",
    "chb01_01.edf",
    "chb01_09.edf",
    "chb01_15.edf",
    "chb01_30.edf",
    "chb01_38.edf",
    "chb01_39.edf",
    "chb01_40.edf",
    "chb01_41.edf"
]


test_recordings = [
    "chb01_18.edf",
    "chb01_21.edf",
    "chb01_26.edf",
    "chb01_42.edf",
    "chb01_46.edf"
]


print("\nTraining recordings:")

for file in train_recordings:

    print(
        " TRAIN:",
        file
    )


print("\nTesting recordings:")

for file in test_recordings:

    print(
        " TEST:",
        file
    )


# ============================================================
# 8. CHECK FOR RECORDING LEAKAGE
# ============================================================

overlap = set(
    train_recordings
).intersection(
    set(test_recordings)
)


print(
    "\nOverlapping recordings:",
    overlap
)


if len(overlap) > 0:

    raise ValueError(
        "Recording leakage detected!"
    )


print(
    "SUCCESS: No recording leakage detected."
)


# ============================================================
# 9. CREATE TRAINING AND TESTING MASKS
# ============================================================

train_mask = (
    metadata_df["file"]
    .isin(train_recordings)
)

test_mask = (
    metadata_df["file"]
    .isin(test_recordings)
)


X_train = X[
    train_mask
]

y_train = y[
    train_mask
]

X_test = X[
    test_mask
]

y_test = y[
    test_mask
]


print("\n" + "=" * 60)
print("RECORDING-WISE DATASET")
print("=" * 60)


print(
    "\nTraining Features Shape:",
    X_train.shape
)

print(
    "Training Labels Shape:",
    y_train.shape
)


print(
    "\nTesting Features Shape:",
    X_test.shape
)

print(
    "Testing Labels Shape:",
    y_test.shape
)


# ============================================================
# 10. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 60)

print(
    y_train.value_counts()
    .sort_index()
)


print("\n" + "=" * 60)
print("TESTING CLASS DISTRIBUTION")
print("=" * 60)

print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# 11. TRAIN CLASS-WEIGHTED RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CLASS-WEIGHTED RANDOM FOREST")
print("=" * 60)


class_weighted_model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


print(
    "\nModel Configuration:"
)

print(
    class_weighted_model
)


print(
    "\nTraining model..."
)


class_weighted_model.fit(

    X_train,

    y_train
)


print(
    "Class-weighted Random Forest training completed."
)


# ============================================================
# 12. SAVE NEW MODEL
# ============================================================

MODEL_DIR.mkdir(
    exist_ok=True
)


joblib.dump(

    class_weighted_model,

    NEW_MODEL_PATH
)


print(
    "\nNew model saved to:"
)

print(
    NEW_MODEL_PATH
)


# ============================================================
# 13. PREDICTION ON UNSEEN RECORDINGS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION ON UNSEEN RECORDINGS")
print("=" * 60)


y_pred = class_weighted_model.predict(

    X_test
)


print(
    "\nPrediction completed."
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred
)


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


print(
    cm
)


# ============================================================
# 15. CONFUSION MATRIX COMPONENTS
# ============================================================

tn, fp, fn, tp = cm.ravel()


print(
    "\nConfusion Matrix Components"
)

print(
    "-" * 60
)

print(
    "True Negatives:",
    tn
)

print(
    "False Positives:",
    fp
)

print(
    "False Negatives:",
    fn
)

print(
    "True Positives:",
    tp
)


# ============================================================
# 16. METRICS
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred
)


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


specificity = (

    tn / (tn + fp)

    if (tn + fp) > 0

    else 0
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
# 17. PRINT METRICS
# ============================================================

print("\n" + "=" * 60)
print("CLASS-IMBALANCE-AWARE MODEL METRICS")
print("=" * 60)


print(
    f"Accuracy:           {accuracy:.4f}"
)


print(
    f"Precision:          {precision:.4f}"
)


print(
    f"Sensitivity/Recall: {sensitivity:.4f}"
)


print(
    f"Specificity:        {specificity:.4f}"
)


print(
    f"F1-Score:           {f1:.4f}"
)


print(
    f"Balanced Accuracy:  {balanced_accuracy:.4f}"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    y_pred,

    target_names=[
        "Normal",
        "Seizure"
    ],

    zero_division=0
)


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


print(
    report
)


# ============================================================
# 19. SAVE CONFUSION MATRIX
# ============================================================

RESULTS_DIR.mkdir(

    exist_ok=True
)


plt.figure(

    figsize=(6, 5)
)


plt.imshow(

    cm
)


plt.title(

    "Class-Weighted Random Forest\n"
    "Recording-Wise Confusion Matrix"
)


plt.xlabel(

    "Predicted Label"
)


plt.ylabel(

    "Actual Label"
)


plt.xticks(

    [0, 1],

    [
        "Normal",
        "Seizure"
    ]
)


plt.yticks(

    [0, 1],

    [
        "Normal",
        "Seizure"
    ]
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


plt.savefig(

    CONFUSION_MATRIX_PATH,

    dpi=300
)


plt.close()


print(
    "\nConfusion matrix saved to:"
)


print(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# 20. INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)


if sensitivity == 0:

    print(
        "\nWARNING:"
    )

    print(
        "The class-weighted Random Forest still failed "
        "to detect seizure windows in unseen recordings."
    )

    print(
        "\nClass weighting alone did not solve the "
        "generalization problem."
    )

else:

    print(
        "\nThe class-weighted Random Forest detected "
        "some seizure windows in unseen recordings."
    )

    print(
        "\nThis indicates that class weighting improved "
        "seizure-class detection compared with the baseline."
    )


print(
    "\nThe results should be compared with:"
)

print(
    "Experiment 01: Random window-level baseline"
)

print(
    "Experiment 02: Recording-wise baseline"
)

print(
    "Experiment 03: Recording-wise class-weighted model"
)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

with open(

    RESULTS_PATH,

    "w"

) as f:

    f.write(
        "EXPERIMENT 03: CLASS-IMBALANCE-AWARE MODELING\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Objective:\n"
    )

    f.write(
        "Investigate whether class weighting improves "
        "seizure detection on unseen EEG recordings.\n\n"
    )

    f.write(
        "Training Recordings:\n"
    )

    for file in train_recordings:

        f.write(
            f"- {file}\n"
        )


    f.write(
        "\nTesting Recordings:\n"
    )

    for file in test_recordings:

        f.write(
            f"- {file}\n"
        )


    f.write(
        "\nDataset Information:\n"
    )

    f.write(
        f"Training Samples: {len(X_train)}\n"
    )

    f.write(
        f"Testing Samples: {len(X_test)}\n"
    )


    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\n"
    )


    f.write(
        "Confusion Matrix Components:\n"
    )

    f.write(
        f"True Negatives: {tn}\n"
    )

    f.write(
        f"False Positives: {fp}\n"
    )

    f.write(
        f"False Negatives: {fn}\n"
    )

    f.write(
        f"True Positives: {tp}\n"
    )


    f.write(
        "\nMetrics:\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Sensitivity/Recall: {sensitivity:.4f}\n"
    )

    f.write(
        f"Specificity: {specificity:.4f}\n"
    )

    f.write(
        f"F1-Score: {f1:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}\n"
    )


    f.write(
        "\nClassification Report:\n"
    )

    f.write(
        report
    )


    f.write(
        "\n\nInterpretation:\n"
    )


    if sensitivity == 0:

        f.write(
            "The class-weighted Random Forest failed "
            "to detect seizure windows in unseen recordings. "
            "Class weighting alone did not solve the "
            "generalization problem.\n"
        )

    else:

        f.write(
            "The class-weighted Random Forest detected "
            "some seizure windows in unseen recordings, "
            "indicating improved seizure-class detection.\n"
        )


print(
    "\nResults saved to:"
)


print(
    RESULTS_PATH
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)

print(
    "EXPERIMENT 03 COMPLETED SUCCESSFULLY"
)

print("=" * 60)