import numpy as np
import pandas as pd
import joblib

from pathlib import Path

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
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("EXPERIMENT 01: BASELINE RANDOM FOREST EVALUATION")
print("=" * 60)

features_df = pd.read_csv(
    DATA_DIR / "features.csv"
)

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

X = features_df[feature_names]

y = features_df["Label"]


print("\nDataset Information")
print("-" * 60)

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)

print("\nClass Distribution:")

print("Normal Windows:",
      np.sum(y == 0))

print("Seizure Windows:",
      np.sum(y == 1))


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTrain-Test Split")
print("-" * 60)

print("Training Samples:",
      len(X_train))

print("Testing Samples:",
      len(X_test))


print("\nTraining Class Distribution:")

print(
    y_train.value_counts()
)


print("\nTesting Class Distribution:")

print(
    y_test.value_counts()
)


# ============================================================
# 3. LOAD BASELINE RANDOM FOREST MODEL
# ============================================================

print("\nLoading Model")
print("-" * 60)

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")

print("Model:",
      type(model).__name__)


# ============================================================
# 4. DEFAULT MODEL PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix")
print("-" * 60)

print(cm)


# ============================================================
# 6. EXTRACT CONFUSION MATRIX VALUES
# ============================================================

tn, fp, fn, tp = cm.ravel()


print("\nConfusion Matrix Components")
print("-" * 60)

print("True Negatives:",
      tn)

print("False Positives:",
      fp)

print("False Negatives:",
      fn)

print("True Positives:",
      tp)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print("-" * 60)

report = classification_report(

    y_test,

    y_pred,

    target_names=[
        "Normal",
        "Seizure"
    ],

    zero_division=0
)

print(report)


# ============================================================
# 8. CALCULATE METRICS
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
# 9. SPECIFICITY
# ============================================================

specificity = (

    tn / (tn + fp)

    if (tn + fp) > 0

    else 0
)


# ============================================================
# 10. FINAL BASELINE METRICS
# ============================================================

print("\nBaseline Model Metrics")
print("-" * 60)

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
# 11. INTERPRETATION
# ============================================================

print("\nInterpretation")
print("-" * 60)

if sensitivity == 0:

    print(
        "WARNING: The baseline model failed to detect "
        "the seizure class in the test set."
    )

    print(
        "This indicates that accuracy alone is misleading "
        "because of severe class imbalance."
    )

else:

    print(
        "The baseline model detected some seizure samples."
    )


# ============================================================
# 12. SAVE EXPERIMENT RESULTS
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    exist_ok=True
)


output_file = (

    RESULTS_DIR

    / "experiment_01_baseline_evaluation.txt"
)


with open(
    output_file,
    "w"
) as f:

    f.write(
        "EXPERIMENT 01: BASELINE RANDOM FOREST EVALUATION\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Feature Shape: {X.shape}\n"
    )

    f.write(
        f"Training Samples: {len(X_train)}\n"
    )

    f.write(
        f"Testing Samples: {len(X_test)}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(
        report
    )

    f.write(
        "\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Sensitivity: {sensitivity:.4f}\n"
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


print("\nExperiment completed successfully.")

print(
    "Results saved to:"
)

print(
    output_file
)