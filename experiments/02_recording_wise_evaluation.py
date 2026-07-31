# ============================================================
# EXPERIMENT 02
# RECORDING-WISE EVALUATION
# ============================================================
#
# Objective:
# Evaluate the trained Random Forest model using a
# recording-wise train-test split.
#
# Important:
# Windows from the same EEG recording are kept together.
# This prevents recording-level data leakage.
#
# This experiment investigates whether the model can
# generalize to EEG recordings that were not used during
# model training.
#
# NOTE:
# This project is a research and learning prototype.
# It is NOT a clinically validated medical diagnostic system.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
# 2. PROJECT PATHS
# ============================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define directories
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create results directory if it does not exist
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Define file paths
FEATURES_PATH = DATA_DIR / "features.csv"

METADATA_PATH = DATA_DIR / "window_metadata.csv"

MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"

RESULT_PATH = (
    RESULTS_DIR /
    "experiment_02_recording_wise_evaluation.txt"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR /
    "experiment_02_recording_wise_confusion_matrix.png"
)


# ============================================================
# 3. PRINT EXPERIMENT INFORMATION
# ============================================================

print("=" * 60)
print("EXPERIMENT 02: RECORDING-WISE EVALUATION")
print("=" * 60)

print("\nProject Root:")
print(PROJECT_ROOT)

print("\nFeatures Path:")
print(FEATURES_PATH)

print("\nMetadata Path:")
print(METADATA_PATH)

print("\nModel Path:")
print(MODEL_PATH)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")
print("-" * 60)

required_files = [
    FEATURES_PATH,
    METADATA_PATH,
    MODEL_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    print(
        f"Found: {file_path.name}"
    )


print("\nAll required files found successfully.")


# ============================================================
# 5. LOAD FEATURE DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING FEATURE DATA")
print("=" * 60)


features_df = pd.read_csv(
    FEATURES_PATH
)

print(
    "\nFeature Dataset Shape:",
    features_df.shape
)


# ============================================================
# 6. LOAD WINDOW METADATA
# ============================================================

metadata_df = pd.read_csv(
    METADATA_PATH
)

print(
    "Metadata Shape:",
    metadata_df.shape
)


# ============================================================
# 7. DISPLAY METADATA INFORMATION
# ============================================================

print("\nMetadata Columns:")

print(
    metadata_df.columns.tolist()
)


# ============================================================
# 8. CHECK DATASET CONSISTENCY
# ============================================================

if len(features_df) != len(metadata_df):

    raise ValueError(
        "Feature rows and metadata rows do not match."
    )


print(
    "\nDataset consistency check passed."
)


# ============================================================
# 9. DEFINE FEATURE COLUMNS
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


# Check that all required feature columns exist

missing_features = [

    feature
    for feature in feature_names
    if feature not in features_df.columns

]


if missing_features:

    raise ValueError(

        "Missing feature columns: "
        + str(missing_features)

    )


# ============================================================
# 10. PREPARE FEATURES AND LABELS
# ============================================================

X = features_df[
    feature_names
].copy()


y = features_df[
    "Label"
].copy()


print("\nFeatures Shape:")
print(X.shape)


print("\nLabels Shape:")
print(y.shape)


# ============================================================
# 11. ADD LABEL TO METADATA
# ============================================================

# The labels in features.csv correspond row-by-row
# with the metadata in window_metadata.csv.

metadata_df = metadata_df.copy()

metadata_df["Label"] = y.values


# ============================================================
# 12. RECORDING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RECORDING SUMMARY")
print("=" * 60)


recording_summary = (

    metadata_df

    .groupby("file")

    .agg(

        total_windows=(
            "Label",
            "count"
        ),

        seizure_windows=(
            "Label",
            "sum"
        )

    )

    .reset_index()

)


recording_summary[
    "normal_windows"
] = (

    recording_summary[
        "total_windows"
    ]

    -

    recording_summary[
        "seizure_windows"
    ]

)


print(
    recording_summary
)


# ============================================================
# 13. IDENTIFY SEIZURE AND NORMAL RECORDINGS
# ============================================================

seizure_recordings = (

    recording_summary[

        recording_summary[
            "seizure_windows"
        ] > 0

    ]

    ["file"]

    .tolist()

)


normal_recordings = (

    recording_summary[

        recording_summary[
            "seizure_windows"
        ] == 0

    ]

    ["file"]

    .tolist()

)


print(
    "\nSeizure-containing recordings:"
)


for file in seizure_recordings:

    print(
        " -",
        file
    )


print(
    "\nNumber of seizure-containing recordings:",
    len(seizure_recordings)
)


print(
    "\nNormal-only recordings:"
)


for file in normal_recordings:

    print(
        " -",
        file
    )


print(
    "\nNumber of normal-only recordings:",
    len(normal_recordings)
)


# ============================================================
# 14. RECORDING-WISE TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
#
# We keep all windows from the same recording together.
#
# Training recordings:
#   4 seizure recordings
#   6 normal recordings
#
# Testing recordings:
#   2 seizure recordings
#   3 normal recordings
#
# This creates:
#   10 training recordings
#   5 testing recordings
#
# No recording appears in both groups.
# ============================================================


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


# ============================================================
# 15. CHECK RECORDING SPLIT
# ============================================================

print("\n" + "=" * 60)
print("RECORDING-WISE TRAIN / TEST SPLIT")
print("=" * 60)


print(
    "\nTraining recordings:",
    len(train_recordings)
)


for file in train_recordings:

    print(
        " TRAIN:",
        file
    )


print(
    "\nTesting recordings:",
    len(test_recordings)
)


for file in test_recordings:

    print(
        " TEST:",
        file
    )


# ============================================================
# 16. CHECK FOR RECORDING LEAKAGE
# ============================================================

overlap = (

    set(train_recordings)

    &

    set(test_recordings)

)


print(
    "\nOverlapping recordings:"
)


print(
    overlap
)


if len(overlap) > 0:

    raise ValueError(

        "Recording leakage detected! "
        "Some recordings appear in both "
        "training and testing sets."

    )


print(
    "\nSUCCESS: No recording leakage detected."
)


# ============================================================
# 17. CREATE TRAINING AND TESTING MASKS
# ============================================================

train_mask = (

    metadata_df[
        "file"
    ]

    .isin(
        train_recordings
    )

)


test_mask = (

    metadata_df[
        "file"
    ]

    .isin(
        test_recordings
    )

)


# ============================================================
# 18. CREATE RECORDING-WISE DATASETS
# ============================================================

X_train_recording = X.loc[
    train_mask
].copy()


y_train_recording = y.loc[
    train_mask
].copy()


X_test_recording = X.loc[
    test_mask
].copy()


y_test_recording = y.loc[
    test_mask
].copy()


print(
    "\n" + "=" * 60
)

print(
    "RECORDING-WISE DATASET"
)

print(
    "=" * 60
)


print(
    "\nTraining Features Shape:",
    X_train_recording.shape
)


print(
    "Training Labels Shape:",
    y_train_recording.shape
)


print(
    "\nTesting Features Shape:",
    X_test_recording.shape
)


print(
    "Testing Labels Shape:",
    y_test_recording.shape
)


# ============================================================
# 19. CLASS DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRAINING CLASS DISTRIBUTION"
)

print(
    "=" * 60
)


print(
    y_train_recording.value_counts()
    .sort_index()
)


print(
    "\n" + "=" * 60
)

print(
    "TESTING CLASS DISTRIBUTION"
)

print(
    "=" * 60
)


print(
    y_test_recording.value_counts()
    .sort_index()
)


# ============================================================
# 20. LOAD TRAINED MODEL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "LOADING TRAINED MODEL"
)

print(
    "=" * 60
)


model = joblib.load(
    MODEL_PATH
)


print(
    "\nModel loaded successfully."
)


print(
    "Model:",
    type(model).__name__
)


# ============================================================
# 21. PREDICTION ON UNSEEN RECORDINGS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "PREDICTION ON UNSEEN RECORDINGS"
)

print(
    "=" * 60
)


y_pred_recording = model.predict(
    X_test_recording
)


print(
    "\nPrediction completed."
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

cm_recording = confusion_matrix(

    y_test_recording,

    y_pred_recording

)


print(
    "\n" + "=" * 60
)

print(
    "CONFUSION MATRIX"
)

print(
    "=" * 60
)


print(
    cm_recording
)


# ============================================================
# 23. CONFUSION MATRIX COMPONENTS
# ============================================================

tn, fp, fn, tp = (

    cm_recording
    .ravel()

)


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
# 24. CALCULATE METRICS
# ============================================================

accuracy_recording = accuracy_score(

    y_test_recording,

    y_pred_recording

)


precision_recording = precision_score(

    y_test_recording,

    y_pred_recording,

    zero_division=0

)


sensitivity_recording = recall_score(

    y_test_recording,

    y_pred_recording,

    zero_division=0

)


specificity_recording = (

    tn / (tn + fp)

    if (tn + fp) > 0

    else 0

)


f1_recording = f1_score(

    y_test_recording,

    y_pred_recording,

    zero_division=0

)


balanced_accuracy_recording = (

    balanced_accuracy_score(

        y_test_recording,

        y_pred_recording

    )

)


# ============================================================
# 25. PRINT FINAL METRICS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "RECORDING-WISE EVALUATION METRICS"
)

print(
    "=" * 60
)


print(
    f"Accuracy:           "
    f"{accuracy_recording:.4f}"
)


print(
    f"Precision:          "
    f"{precision_recording:.4f}"
)


print(
    f"Sensitivity/Recall: "
    f"{sensitivity_recording:.4f}"
)


print(
    f"Specificity:        "
    f"{specificity_recording:.4f}"
)


print(
    f"F1-Score:           "
    f"{f1_recording:.4f}"
)


print(
    f"Balanced Accuracy:  "
    f"{balanced_accuracy_recording:.4f}"
)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

recording_report = classification_report(

    y_test_recording,

    y_pred_recording,

    target_names=[
        "Normal",
        "Seizure"
    ],

    zero_division=0

)


print(
    "\n" + "=" * 60
)

print(
    "RECORDING-WISE CLASSIFICATION REPORT"
)

print(
    "=" * 60
)


print(
    recording_report
)


# ============================================================
# 27. SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(

    figsize=(
        6,
        5
    )

)


plt.imshow(
    cm_recording
)


plt.title(
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

            cm_recording[
                i,
                j
            ],

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
# 28. INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "INTERPRETATION"
)

print(
    "=" * 60
)


if sensitivity_recording == 0:

    print(

        "\nWARNING: The model failed to detect "
        "the seizure class in the unseen "
        "recordings."

    )

    print(

        "\nThis indicates that the model's "
        "high accuracy is largely influenced "
        "by the severe class imbalance."

    )

    print(

        "\nAccuracy alone is therefore not "
        "sufficient for evaluating this "
        "EEG seizure detection model."

    )

else:

    print(

        "\nThe model detected some seizure "
        "windows in the unseen recordings."

    )


# ============================================================
# 29. SAVE EXPERIMENT RESULTS
# ============================================================

with open(

    RESULT_PATH,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "EXPERIMENT 02: RECORDING-WISE EVALUATION\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )


    f.write(
        "Objective:\n"
    )

    f.write(

        "Evaluate the trained Random Forest "
        "model using a recording-wise split "
        "to investigate generalization to "
        "unseen EEG recordings.\n\n"

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
        "\nRecording Leakage:\n"
    )

    f.write(
        f"{overlap}\n"
    )


    f.write(
        "\nDataset Information:\n"
    )

    f.write(
        f"Training Samples: "
        f"{len(X_train_recording)}\n"
    )

    f.write(
        f"Testing Samples: "
        f"{len(X_test_recording)}\n"
    )


    f.write(
        "\nTraining Class Distribution:\n"
    )

    f.write(
        str(
            y_train_recording
            .value_counts()
            .sort_index()
        )

    )


    f.write(
        "\n\nTesting Class Distribution:\n"
    )

    f.write(
        str(
            y_test_recording
            .value_counts()
            .sort_index()
        )

    )


    f.write(
        "\n\nConfusion Matrix:\n"
    )

    f.write(
        str(
            cm_recording
        )
    )


    f.write(
        "\n\nConfusion Matrix Components:\n"
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
        "\nClassification Report:\n"
    )

    f.write(
        recording_report
    )


    f.write(
        "\n\nFinal Metrics:\n"
    )

    f.write(
        f"Accuracy: "
        f"{accuracy_recording:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{precision_recording:.4f}\n"
    )

    f.write(
        f"Sensitivity/Recall: "
        f"{sensitivity_recording:.4f}\n"
    )

    f.write(
        f"Specificity: "
        f"{specificity_recording:.4f}\n"
    )

    f.write(
        f"F1-Score: "
        f"{f1_recording:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_recording:.4f}\n"
    )


    f.write(
        "\n\nResearch Interpretation:\n"
    )

    f.write(

        "The recording-wise evaluation "
        "prevents windows from the same "
        "EEG recording from appearing in "
        "both training and testing sets.\n"

    )

    f.write(

        "The model was evaluated on recordings "
        "that were not used during training.\n"

    )

    f.write(

        "If seizure sensitivity is low despite "
        "high accuracy, the result indicates "
        "that severe class imbalance is "
        "affecting seizure detection performance.\n"

    )

    f.write(

        "Accuracy alone should therefore not "
        "be considered sufficient for assessing "
        "EEG seizure detection performance.\n"

    )


print(
    "\nResults saved to:"
)


print(
    RESULT_PATH
)


# ============================================================
# 30. FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "EXPERIMENT 02 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)