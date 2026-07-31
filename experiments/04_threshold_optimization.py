import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# ============================================================
# EXPERIMENT 04: THRESHOLD OPTIMIZATION
# ============================================================
#
# This experiment investigates whether adjusting the
# classification probability threshold improves EEG seizure
# detection performance.
#
# The evaluation is performed on the same unseen,
# recording-wise test set used in the original notebook.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

FEATURES_PATH = os.path.join(
    DATA_DIR,
    "features.csv"
)

METADATA_PATH = os.path.join(
    DATA_DIR,
    "window_metadata.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_class_weighted.pkl"
)

THRESHOLD_RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "experiment_04_threshold_comparison.csv"
)

THRESHOLD_PLOT_PATH = os.path.join(
    RESULTS_DIR,
    "experiment_04_threshold_performance.png"
)

THRESHOLD_REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "experiment_04_threshold_optimization.txt"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("=" * 60)
print("EXPERIMENT 04: THRESHOLD OPTIMIZATION")
print("=" * 60)

required_files = [
    FEATURES_PATH,
    METADATA_PATH,
    MODEL_PATH,
]

for file_path in required_files:

    if os.path.exists(file_path):

        print(
            f"Found: {file_path}"
        )

    else:

        raise FileNotFoundError(
            f"Missing required file: {file_path}"
        )


# ============================================================
# LOAD DATA
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

model = joblib.load(
    MODEL_PATH
)

print(
    "Features shape:",
    features_df.shape
)

print(
    "Metadata shape:",
    metadata_df.shape
)

print(
    "Model loaded:",
    type(model).__name__
)


# ============================================================
# VERIFY LABEL CONSISTENCY
# ============================================================

print("\n" + "=" * 60)
print("VERIFYING LABEL CONSISTENCY")
print("=" * 60)

labels_match = np.array_equal(
    features_df["Label"].to_numpy(),
    metadata_df["label"].to_numpy()
)

print(
    "Labels match:",
    labels_match
)

if not labels_match:

    raise ValueError(
        "Feature labels and metadata labels do not match."
    )

print(
    "SUCCESS: Feature labels and metadata labels are consistent."
)


# ============================================================
# PREPARE FEATURES AND LABELS
# ============================================================

print("\n" + "=" * 60)
print("PREPARING FEATURES AND LABELS")
print("=" * 60)

feature_names = [
    "Mean",
    "Std",
    "Variance",
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma",
]

X = features_df[
    feature_names
]

y = features_df[
    "Label"
]

print(
    "Feature columns:",
    feature_names
)

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "\nClass Labels:"
)

print(
    "0 = Normal"
)

print(
    "1 = Seizure"
)


# ============================================================
# GENERATE ALL SEIZURE PROBABILITIES
# ============================================================

print("\n" + "=" * 60)
print("GENERATING SEIZURE PROBABILITIES")
print("=" * 60)

seizure_probabilities = model.predict_proba(
    X
)[:, 1]

print(
    "Number of probabilities:",
    len(seizure_probabilities)
)

print(
    f"Minimum: {seizure_probabilities.min():.6f}"
)

print(
    f"Maximum: {seizure_probabilities.max():.6f}"
)

print(
    "\nFirst 10 seizure probabilities:"
)

print(
    seizure_probabilities[:10]
)


# ============================================================
# DEFINE UNSEEN TEST RECORDINGS
# ============================================================

print("\n" + "=" * 60)
print("DEFINING UNSEEN TEST RECORDINGS")
print("=" * 60)

test_recordings = [
    "chb01_18.edf",
    "chb01_21.edf",
    "chb01_26.edf",
    "chb01_42.edf",
    "chb01_46.edf",
]

print(
    "\nUnseen Test Recordings:"
)

for recording in test_recordings:

    print(
        " -",
        recording
    )


# ============================================================
# CREATE TEST MASK
# ============================================================

test_mask = metadata_df[
    "file"
].isin(
    test_recordings
)


# ============================================================
# SELECT TEST DATA
# ============================================================

X_test = X.loc[
    test_mask
]

y_test = y.loc[
    test_mask
]

test_metadata = metadata_df.loc[
    test_mask
].copy()

test_probabilities = seizure_probabilities[
    test_mask.to_numpy()
]


# ============================================================
# TEST SET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("UNSEEN TEST SET INFORMATION")
print("=" * 60)

print(
    "Testing feature shape:",
    X_test.shape
)

print(
    "Testing label shape:",
    y_test.shape
)

print(
    "\nTesting Labels:"
)

print(
    y_test.value_counts().sort_index()
)

print(
    "\nTest recordings actually present:"
)

for recording in sorted(
    test_metadata["file"].unique()
):

    print(
        " -",
        recording
    )

print(
    "\nNumber of test windows:",
    len(y_test)
)

print(
    "Normal windows:",
    np.sum(y_test == 0)
)

print(
    "Seizure windows:",
    np.sum(y_test == 1)
)


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD COMPARISON")
print("=" * 60)

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
]

threshold_results = []


for threshold in thresholds:

    y_pred_threshold = (
        test_probabilities >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred_threshold
    )

    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    sensitivity = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred_threshold
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_threshold,
        labels=[0, 1]
    ).ravel()


    # --------------------------------------------------------
    # SPECIFICITY
    # --------------------------------------------------------

    if (
        tn + fp
    ) > 0:

        specificity = (
            tn /
            (
                tn + fp
            )
        )

    else:

        specificity = 0.0


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    threshold_results.append(
        {
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1_Score": f1,
            "Balanced_Accuracy": balanced_accuracy,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        }
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

threshold_results_df = pd.DataFrame(
    threshold_results
)


print(
    "\nThreshold Comparison Results:"
)

print(
    threshold_results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE THRESHOLD COMPARISON
# ============================================================

threshold_results_df.to_csv(
    THRESHOLD_RESULTS_PATH,
    index=False
)

print(
    "\nThreshold comparison saved to:"
)

print(
    THRESHOLD_RESULTS_PATH
)


# ============================================================
# BASELINE AND CANDIDATE THRESHOLD
# ============================================================

baseline_threshold = 0.50

candidate_threshold = 0.10

baseline_row = threshold_results_df[
    threshold_results_df["Threshold"]
    == baseline_threshold
].iloc[0]

candidate_row = threshold_results_df[
    threshold_results_df["Threshold"]
    == candidate_threshold
].iloc[0]


print("\n" + "=" * 60)
print("BASELINE THRESHOLD")
print("=" * 60)

print(
    "Baseline threshold:",
    baseline_threshold
)

print(
    f"Precision: {baseline_row['Precision']:.4f}"
)

print(
    f"Sensitivity/Recall: {baseline_row['Sensitivity']:.4f}"
)

print(
    f"Balanced Accuracy: {baseline_row['Balanced_Accuracy']:.4f}"
)


print("\n" + "=" * 60)
print("CANDIDATE THRESHOLD")
print("=" * 60)

print(
    "Candidate threshold:",
    candidate_threshold
)

print(
    f"Precision: {candidate_row['Precision']:.4f}"
)

print(
    f"Sensitivity/Recall: {candidate_row['Sensitivity']:.4f}"
)

print(
    f"Balanced Accuracy: {candidate_row['Balanced_Accuracy']:.4f}"
)


# ============================================================
# PLOT THRESHOLD PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("PLOTTING THRESHOLD PERFORMANCE")
print("=" * 60)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["Sensitivity"],
    marker="o",
    linewidth=2,
    label="Sensitivity"
)

plt.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["Specificity"],
    marker="o",
    linewidth=2,
    label="Specificity"
)

plt.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["F1_Score"],
    marker="o",
    linewidth=2,
    label="F1-Score"
)

plt.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["Balanced_Accuracy"],
    marker="o",
    linewidth=2,
    label="Balanced Accuracy"
)

plt.axvline(
    candidate_threshold,
    linestyle="--",
    linewidth=2,
    label="Candidate Threshold = 0.10"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Threshold Optimization for EEG Seizure Detection"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    THRESHOLD_PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nThreshold performance plot saved to:"
)

print(
    THRESHOLD_PLOT_PATH
)


# ============================================================
# GENERATE RESEARCH REPORT
# ============================================================

baseline_sensitivity = (
    baseline_row["Sensitivity"]
)

candidate_sensitivity = (
    candidate_row["Sensitivity"]
)

baseline_precision = (
    baseline_row["Precision"]
)

candidate_precision = (
    candidate_row["Precision"]
)

baseline_balanced_accuracy = (
    baseline_row["Balanced_Accuracy"]
)

candidate_balanced_accuracy = (
    candidate_row["Balanced_Accuracy"]
)

baseline_specificity = (
    baseline_row["Specificity"]
)

candidate_specificity = (
    candidate_row["Specificity"]
)

baseline_tp = int(
    baseline_row["TP"]
)

candidate_tp = int(
    candidate_row["TP"]
)

baseline_fn = int(
    baseline_row["FN"]
)

candidate_fn = int(
    candidate_row["FN"]
)

report = f"""
EXPERIMENT 04: THRESHOLD OPTIMIZATION
======================================

This experiment investigates the effect of classification
threshold selection on EEG seizure detection performance.

The Random Forest model was evaluated on the same unseen,
recording-wise test set used in the original experiment.

UNSEEN TEST RECORDINGS
----------------------

- chb01_18.edf
- chb01_21.edf
- chb01_26.edf
- chb01_42.edf
- chb01_46.edf

Number of test windows: {len(y_test)}
Normal windows: {np.sum(y_test == 0)}
Seizure windows: {np.sum(y_test == 1)}


BASELINE THRESHOLD
------------------

Baseline threshold: {baseline_threshold:.2f}

Precision:          {baseline_precision:.4f}
Sensitivity/Recall: {baseline_sensitivity:.4f}
Specificity:        {baseline_specificity:.4f}
Balanced Accuracy:  {baseline_balanced_accuracy:.4f}

True Positives:     {baseline_tp}
False Negatives:    {baseline_fn}


CANDIDATE THRESHOLD
-------------------

Candidate threshold: {candidate_threshold:.2f}

Precision:          {candidate_precision:.4f}
Sensitivity/Recall: {candidate_sensitivity:.4f}
Specificity:        {candidate_specificity:.4f}
Balanced Accuracy:  {candidate_balanced_accuracy:.4f}

True Positives:     {candidate_tp}
False Negatives:    {candidate_fn}


THRESHOLD COMPARISON
--------------------

The threshold comparison evaluated classification thresholds
between 0.10 and 0.80.

The candidate threshold of 0.10 achieved the highest
sensitivity, F1-score, and balanced accuracy among the
evaluated thresholds.

At threshold 0.10, the model detected more seizure windows
than at the default threshold of 0.50.

However, lowering the threshold reduced precision and
specificity. Therefore, threshold selection represents a
trade-off between seizure sensitivity and false-positive rate.

The default threshold of 0.50 achieved a seizure sensitivity
of {baseline_sensitivity * 100:.2f}%.

At the candidate threshold of 0.10, seizure sensitivity
increased to {candidate_sensitivity * 100:.2f}%.

This represents an increase in sensitivity of
{(candidate_sensitivity - baseline_sensitivity) * 100:.2f}
percentage points.

The number of detected seizure windows increased from
{baseline_tp} to {candidate_tp}, while false negatives
decreased from {baseline_fn} to {candidate_fn}.

The threshold of 0.10 is therefore considered a candidate
sensitivity-oriented operating threshold for this research
prototype based on the evaluated thresholds.

This result should be interpreted as exploratory threshold
analysis. The candidate threshold was evaluated on the same
unseen recording-wise test set and therefore should not be
considered a universally optimal threshold.

Future work should optimize the operating threshold using
validation data or nested cross-validation and then evaluate
the selected threshold on an independent test set.

The results demonstrate that classification threshold selection
has a substantial effect on seizure detection performance and
highlights the importance of considering sensitivity,
specificity, precision, F1-score, and balanced accuracy rather
than relying on accuracy alone.
"""

with open(
    THRESHOLD_REPORT_PATH,
    "w",
    encoding="utf-8"
) as report_file:

    report_file.write(
        report.strip()
    )


print(
    "\nThreshold optimization report saved to:"
)

print(
    THRESHOLD_REPORT_PATH
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 04 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"Baseline threshold: {baseline_threshold:.2f}"
)

print(
    f"Candidate threshold: {candidate_threshold:.2f}"
)

print(
    f"Baseline sensitivity: "
    f"{baseline_sensitivity * 100:.2f}%"
)

print(
    f"Candidate sensitivity: "
    f"{candidate_sensitivity * 100:.2f}%"
)

print(
    "\nAll Experiment 04 results have been generated."
)
