import os
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Load threshold optimization results
threshold_file = os.path.join(
    RESULTS_DIR,
    "experiment_04_threshold_comparison.csv"
)

threshold_df = pd.read_csv(threshold_file)

print("Threshold results loaded successfully.")
print(threshold_df)

# ============================================================
# FIGURE 1: THRESHOLD PERFORMANCE TRADE-OFF
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Sensitivity"] * 100,
    marker="o",
    linewidth=2,
    label="Sensitivity"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"] * 100,
    marker="o",
    linewidth=2,
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Balanced_Accuracy"] * 100,
    marker="o",
    linewidth=2,
    label="Balanced Accuracy"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Performance (%)")
plt.title("Threshold Optimization: Sensitivity-Precision Trade-off")

plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_file = os.path.join(
    RESULTS_DIR,
    "final_threshold_optimization_tradeoff.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Figure 1 saved successfully: {output_file}"
)


# ============================================================
# FIGURE 2: RECORDING-WISE CROSS-VALIDATION COMPARISON
# ============================================================

cv_data = pd.DataFrame({
    "Metric": [
        "Sensitivity",
        "Precision",
        "Specificity",
        "F1 Score",
        "Balanced Accuracy"
    ],
    "Threshold 0.5": [
        0.4752,
        0.7619,
        0.9989,
        0.5854,
        0.7371
    ],
    "Threshold 0.1": [
        0.7327,
        0.5175,
        0.9947,
        0.6066,
        0.8637
    ]
})

x = range(len(cv_data["Metric"]))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    cv_data["Threshold 0.5"] * 100,
    width=width,
    label="Threshold 0.5"
)

plt.bar(
    [i + width / 2 for i in x],
    cv_data["Threshold 0.1"] * 100,
    width=width,
    label="Threshold 0.1"
)

plt.xticks(
    list(x),
    cv_data["Metric"]
)

plt.xlabel("Performance Metric")
plt.ylabel("Performance (%)")

plt.title(
    "5-Fold Recording-Wise Cross-Validation: Threshold Comparison"
)

plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()

output_file = os.path.join(
    RESULTS_DIR,
    "final_recording_wise_cross_validation_comparison.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Figure 2 saved successfully: {output_file}"
)

