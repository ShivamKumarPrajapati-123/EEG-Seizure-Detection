🧠 EEG-Based Epileptic Seizure Detection Using Signal Processing and Machine Learning

🚀 **Live Demo:** https://shivam-eeg-seizure-detector.streamlit.app

Author: Shivam Prajapati
Program: Computer Science Engineering --- Artificial Intelligence &
Machine Learning

1. Project Overview

Epileptic seizures are neurological events that can produce abnormal
patterns in electroencephalography (EEG) signals. Automatic seizure
detection from EEG recordings is an important application of biomedical
signal processing and machine learning.

This project presents an end-to-end research and learning pipeline for
analyzing EEG recordings and detecting seizure-related activity using
signal processing, feature engineering, and machine learning.

The project uses the CHB-MIT Scalp EEG Database and processes EEG
recordings stored in EDF (European Data Format) files.

The project covers:

EEG data loading using MNE-Python

EEG metadata and signal inspection

Signal preprocessing

Fixed-length EEG windowing

Seizure/non-seizure labeling

Statistical feature extraction

Frequency-domain feature extraction using Welch PSD

Random Forest classification

Class-imbalance handling

Recording-wise evaluation

Threshold analysis

Recording-wise cross-validation

Event-level evaluation

Temporal consecutive-window analysis

Cross-patient evaluation

Reproducible research artifacts

Streamlit-based deployment

Important: This is an educational and research prototype. It is
not a clinically validated medical diagnostic system and must not
be used for medical diagnosis or clinical decision-making.

2. Objectives

The main objectives are to:

Load EEG recordings from EDF files.

Understand EEG signal properties and metadata.

Preprocess EEG signals using frequency filtering.

Segment continuous EEG recordings into fixed-length windows.

Identify seizure and non-seizure EEG windows.

Extract statistical and frequency-domain features.

Construct a machine-learning feature dataset.

Train a Random Forest classifier.

Handle severe class imbalance using class-weighted learning.

Evaluate performance using appropriate metrics.

Investigate classification-threshold behavior.

Evaluate generalization to unseen recordings.

Evaluate seizure detection at the event level.

Investigate temporal consecutive-window post-processing.

Evaluate cross-patient generalization.

Produce reproducible results, figures, and research artifacts.

Deploy the prediction pipeline using Streamlit.

3. Dataset

This project uses the CHB-MIT Scalp EEG Database, a publicly
available long-term scalp EEG dataset containing recordings from
pediatric subjects with intractable seizures.

The EEG recordings are provided as EDF files and are processed using
MNE-Python.

Raw EDF recordings are not included in this GitHub repository because of
their size. They must be obtained separately from the CHB-MIT database.

CHB01 robustness dataset

The original recording-wise robustness experiments used 15 CHB01
recordings:

Parameter                          Value

EEG recordings                        15
Seizure-containing recordings          6
Normal-only recordings                 9
EEG windows                       13,181
Normal windows                    13,080
Seizure windows                      101
Features                               8

This dataset is severely imbalanced, so accuracy alone is not sufficient
for interpretation.

Multi-patient evaluation

The later cross-patient experiments evaluated a model trained on
CHB01 against two previously unseen patients:

Training patient: CHB01
Test patient 1:   CHB02
Test patient 2:   CHB03

The CHB02 and CHB03 experiments were kept separate from threshold
tuning. The threshold used in both final cross-patient experiments was
selected only from CHB01 internal validation.

4. Complete System Workflow

Raw EEG EDF Recordings
        │
        ▼
EEG Loading using MNE
        │
        ▼
Metadata / Signal Inspection
        │
        ▼
Preprocessing
        │
        ▼
4-Second Windowing
        │
        ▼
Seizure / Non-Seizure Labeling
        │
        ▼
Feature Extraction
        │
        ├──────────────────────┐
        ▼                      ▼
Statistical Features    Welch PSD Features
        │                      │
        └───────────┬──────────┘
                    ▼
          Machine-Learning Dataset
                    │
                    ▼
       Class-Weighted Random Forest
                    │
                    ▼
          Probability Predictions
                    │
        ┌───────────┴──────────────┐
        ▼                          ▼
 Threshold / Validation     Cross-Patient Testing
        │                    CHB01 → CHB02
        │                    CHB01 → CHB03
        ▼                          │
 Event-Level Analysis              │
        │                          │
        ▼                          ▼
Temporal Analysis           Final Frozen Results
        │
        ▼
Research Interpretation
        │
        ▼
Streamlit Deployment

5. EEG Signal Preprocessing

Raw EEG signals are loaded from EDF files using MNE-Python.

The current preprocessing pipeline uses a band-pass range of:

Parameter        Value

Low cutoff      0.5 Hz
High cutoff      40 Hz

The preprocessing stage reduces unwanted low-frequency drift and
high-frequency components before windowing and feature extraction.

MNE may report duplicate channel names such as T8-P8; when this
occurs, MNE automatically renames duplicate channels to maintain unique
channel identifiers.

6. EEG Windowing

The prediction pipeline uses non-overlapping fixed-length windows.

Parameter                  Value

Sampling frequency        256 Hz
Window duration        4 seconds
Samples per window         1,024

For a 23-channel recording, an individual window has the representation:

23 channels × 1024 samples

A 3,600-second recording sampled at 256 Hz contains:

3,600 / 4 = 900 complete windows

Each window is converted into a feature vector before machine-learning
prediction.

7. Seizure Labeling

Each EEG window receives a binary label:

0 → Normal / Non-Seizure
1 → Seizure

Seizure annotations associated with the recordings are used to determine
whether a window overlaps a seizure interval.

The original CHB01 robustness dataset contained:

13,181 total windows
13,080 normal windows
101 seizure windows

The strong class imbalance is an important consideration throughout
model evaluation.

8. Feature Extraction

The final feature representation contains eight features.

Statistical features

Mean

Standard deviation

Variance

Frequency-domain features

Power Spectral Density (PSD) is estimated using the Welch method.

Band      Frequency range

Delta           0.5--4 Hz
Theta             4--8 Hz
Alpha            8--13 Hz
Beta            13--30 Hz
Gamma           30--40 Hz

The final feature vector is:

Mean
Std
Variance
Delta
Theta
Alpha
Beta
Gamma

The implementation extracts statistics from the window and computes
channel-averaged Welch band-power features.

9. Machine Learning Model

The primary classifier is a Random Forest Classifier.

The final frozen cross-patient model uses:

RandomForestClassifier
n_estimators = 200
class_weight = "balanced"

The model provides class probabilities, allowing a probability threshold
to be applied during seizure classification.

Model artifacts

Two model artifacts are present in the repository:

- `models/random_forest_model.pkl` — earlier/baseline model artifact from the initial experiments.
- `models/phase2_class_weighted_recording_wise.pkl` — final frozen class-weighted Random Forest model used for the CHB01 → CHB02/CHB03 cross-patient experiments.

The class-weighted recording-wise model is the model used for the
final frozen CHB01 → CHB02/CHB03 cross-patient experiments.

10. Evaluation Strategy

The project evaluates performance at several levels.

Window-level evaluation

Metrics include:

Accuracy

Precision

Sensitivity/Recall

Specificity

F1-score

Balanced accuracy

Confusion matrix

ROC-AUC

Average Precision

Recording-wise evaluation

Recordings are kept separate when evaluating generalization to unseen
recordings.

This reduces the risk of overly optimistic results caused by highly
correlated windows from the same EEG recording appearing in both
training and testing.

Event-level evaluation

Predictions are also grouped into seizure events to determine whether an
actual seizure event was detected.

Temporal evaluation

Consecutive positive windows are investigated as a simple temporal
consistency strategy.

11. Experimental Progression

The project was developed through multiple experimental stages.

Experiment 03 --- Recording-Wise Holdout

An early recording-wise holdout experiment evaluated the model at a
threshold of 0.50.

Metric                Result

Accuracy              98.73%
Precision             83.33%
Sensitivity           34.25%
Specificity           99.88%
F1-score              48.54%
Balanced accuracy     67.06%

This experiment demonstrated high specificity but limited seizure
sensitivity.

Experiment 04 --- Exploratory Threshold Optimization

An exploratory threshold experiment investigated lower classification
thresholds.

A threshold of:

0.10

was selected in that experiment.

Metric                Result

Accuracy              98.97%
Precision             65.62%
Sensitivity           86.30%
Specificity           99.20%
F1-score              74.56%
Balanced accuracy     92.75%

The experiment demonstrated the sensitivity/precision trade-off produced
by changing the probability threshold.

Experiment 05 --- Recording/Event-Level Holdout

A small recording/event-level evaluation produced:

Metric           Result

Accuracy         60.00%
Precision        60.00%
Sensitivity     100.00%
Specificity       0.00%
F1-score         75.00%

Because the evaluation set was very small, these results are
exploratory.

Experiment 06 --- Recording-Wise 5-Fold Cross-Validation

The project then used recording-wise cross-validation with recording
filename as the grouping variable.

The earlier robustness dataset contained:

15 recordings
6 seizure-containing recordings
9 normal-only recordings
13,181 windows

This protocol provides a stronger evaluation of generalization to unseen
recordings than random window-level splitting.

However, this experiment does not by itself establish generalization to
completely unseen patients.

12. Recording-Wise Cross-Validation Results

Two thresholds were examined using the cross-validation predictions.

Threshold 0.50

Metric                Result

Accuracy              99.48%
Precision             76.19%
Sensitivity           47.52%
Specificity           99.89%
F1-score              58.54%
Balanced accuracy     73.71%

Pooled confusion matrix:

TN = 13,065
FP = 15
FN = 53
TP = 48

Threshold 0.10

Metric                Result

Accuracy              99.27%
Precision             51.75%
Sensitivity           73.27%
Specificity           99.47%
F1-score              60.66%
Balanced accuracy     86.37%

Pooled confusion matrix:

TN = 13,011
FP = 69
FN = 27
TP = 74

The threshold comparison demonstrates a clear operating-point trade-off:

Sensitivity:       47.52% → 73.27%
Balanced accuracy: 73.71% → 86.37%
Precision:         76.19% → 51.75%

The lower threshold increased seizure sensitivity while also increasing
false-positive predictions.

13. Threshold Selection Limitation

The threshold of 0.10 originated from the exploratory threshold
analysis.

For the later final cross-patient experiments, the operating threshold
was not selected from CHB02 or CHB03.

Instead:

Threshold = 0.05

was selected using CHB01 internal validation and then frozen.

CHB02 and CHB03 were used only for final evaluation.

This distinction is important because tuning the threshold directly on a
test patient would introduce test-set leakage.

A future nested validation design can provide an even more rigorous
threshold-selection procedure.

14. Final Frozen Cross-Patient Experiment

The strongest patient-generalization experiment currently included in
the repository is:

Training patient → CHB01
Test patient      → CHB02
Test patient      → CHB03

The frozen configuration is:

Parameter                                  Value

Training patient                           CHB01
Test patients                       CHB02, CHB03
Sampling frequency                        256 Hz
Window duration                            4 sec
Window size                        1,024 samples
Features                                       8
Model                     RandomForestClassifier
Estimators                                   200
Class weight                            balanced
Threshold                                   0.05
Threshold source       CHB01 internal validation
CHB02 tuning                                  No
CHB03 tuning                                  No
Status                       Complete and frozen

The frozen model was not retrained on CHB02 or CHB03.

15. CHB02 Cross-Patient Results

CHB02 was evaluated using the frozen CHB01-trained model and threshold
0.05.

Metric                                CHB02

Test windows                          6,539
Actual seizure windows                   45
Predicted seizure windows               385
Accuracy                             94.34%
Sensitivity                          66.67%
Specificity                          94.53%
Precision                             7.79%
F1-score                             13.95%
ROC-AUC                              93.08%
Average Precision                    16.66%
Actual seizure events                     3
Detected seizure events                   3
Missed seizure events                     0
Time-weighted seizure coverage       66.28%
Average detection delay            3.33 sec
Predicted events                        201
False-positive events                   195
False-positive events/hour            37.03

The ROC-AUC value is a threshold-independent ranking metric and should
not be interpreted as 93.08% classification accuracy.

The event-level analysis detected all three annotated seizure events in
this evaluation, but the large number of false-positive events shows
that event detection alone does not fully describe the system's
behavior.

16. CHB03 Cross-Patient Results

CHB03 was evaluated using the same frozen CHB01-trained model and
threshold 0.05.

Metric                                CHB03

Test windows                         34,201
Actual seizure windows                  106
Predicted seizure windows               537
Accuracy                             98.28%
Sensitivity                          25.47%
Specificity                          98.50%
Precision                             5.03%
F1-score                              8.40%
ROC-AUC                              86.58%
Average Precision                     4.35%
Actual seizure events                     7
Detected seizure events                   3
Missed seizure events                     4
Time-weighted seizure coverage       26.37%
Average detection delay            9.67 sec
Predicted events                        224
False-positive events                   216
False-positive events/hour             5.68

Again, ROC-AUC should not be interpreted as accuracy.

The CHB03 results show lower seizure sensitivity and lower seizure
coverage than CHB02 under the same frozen model and threshold.

17. CHB02 vs CHB03 Cross-Patient Comparison

Metric                            CHB02      CHB03

Test windows                      6,539     34,201
Actual seizure events                 3          7
Detected events                       3          3
Sensitivity                      66.67%     25.47%
Specificity                      94.53%     98.50%
Precision                         7.79%      5.03%
F1-score                         13.95%      8.40%
ROC-AUC                          93.08%     86.58%
Average Precision                16.66%      4.35%
Seizure coverage                 66.28%     26.37%
Detection delay                3.33 sec   9.67 sec
False-positive events/hour        37.03       5.68

The comparison demonstrates variability in cross-patient performance.

The lower false-positive event rate observed for CHB03 occurred together
with lower sensitivity, lower seizure coverage, and longer detection
delay. Therefore, it should not be described as an overall performance
improvement.

The relative differences are descriptive comparisons, not statistical
significance tests.

18. Observed Inter-Patient Feature Differences

Feature distributions were also examined across patients.

Examples of observed univariate separability in the analysis include:

CHB02

Theta    ≈ 0.985
Std      ≈ 0.958
Alpha    ≈ 0.956
Variance ≈ 0.945
Delta    ≈ 0.934
Beta     ≈ 0.831
Gamma    ≈ 0.669
Mean     ≈ 0.512

CHB01

Theta    ≈ 0.989
Std      ≈ 0.984
Variance ≈ 0.979
Delta    ≈ 0.970
Alpha    ≈ 0.904
Beta     ≈ 0.853
Gamma    ≈ 0.779
Mean     ≈ 0.542 separability AUC

Distribution and KS analyses also showed differences in several feature
distributions between patients.

These findings are described as observed inter-patient feature
distribution differences. They do not establish a causal explanation
for the change in model performance.

19. Event-Level and Temporal Analysis

Earlier integrated experiments investigated whether temporal consistency
could reduce isolated false-positive predictions.

The analysis examined configurations requiring multiple consecutive
positive 4-second windows.

Consecutive            Time   Accuracy   Precision   Sensitivity   Specificity         F1
positive        requirement
windows

1                     4 sec        60%         60%          100%            0%        75%

2                     8 sec        80%         75%          100%           50%     85.71%

3                    12 sec        80%         75%          100%           50%     85.71%

5                    20 sec       100%        100%          100%          100%       100%

These results came from a small evaluation set containing three
seizure-containing recordings and two normal-only recordings.

Therefore, the apparent 100% sensitivity and 100% specificity at the
20-second configuration are exploratory observations, not evidence
of clinical performance.

20. Research Artifacts

The repository contains reproducible research outputs for the
cross-patient experiments.

Important result files include:

results/chb02_cross_patient_final_results.csv
results/chb03_cross_patient_final_results.csv
results/chb01_cross_patient_master_results.csv
results/chb01_cross_patient_research_summary.csv
results/chb02_vs_chb03_performance_change.csv
results/chb01_cross_patient_experiment_metadata.json
results/chb02_experiment_frozen.txt
results/chb03_experiment_frozen.txt

Important figures include:

images/chb01_to_chb02_roc_curve.png
images/chb01_to_chb02_precision_recall_curve.png
images/chb01_to_chb02_calibration_curve.png
images/chb01_to_chb02_final_confusion_matrix.png
images/chb01_to_chb03_roc_curve.png
images/chb01_to_chb03_precision_recall_curve.png
images/chb02_vs_chb03_performance_comparison.png
images/chb02_vs_chb03_performance_comparison.svg
images/chb02_vs_chb03_performance_change.svg
images/chb01_cross_patient_research_summary.svg

The final research package audit confirmed the required cross-patient
artifacts are present.

21. Notebook Organization

The repository currently contains 22 notebooks:

01_Reading_EEG.ipynb
02_Understanding_EEG.ipynb
03_Preprocessing.ipynb
04_Windowing.ipynb
05_Labeling.ipynb
06_Feature_Extraction.ipynb
07_Model_Training.ipynb
08_Dataset_Builder.ipynb
09_Model_Evaluation.ipynb
10_Prediction.ipynb
11_Recording_Wise_Evaluation.ipynb
12_Seizure_Event_Level_Evaluation.ipynb
13_Robustness_Cross_Validation.ipynb
14_Final_Integrated_Evaluation.ipynb
15_Final_Temporal_Event_Evaluation.ipynb
16_Threshold_Optimization.ipynb
17_Leakage_Free_Final_Experiment.ipynb
18_Final_Figures.ipynb
19._Final_Technical_Audit.ipynb
20_Multi_Patient_Dataset_Builder.ipynb
21_CHB02_Cross_Patient_Test.ipynb
22_CHB03_Cross_Patient_Test.ipynb

The notebooks document the progression from basic EEG inspection through
preprocessing, feature extraction, modeling, robustness analysis,
multi-patient evaluation, and final research auditing.

22. Project Structure

EEG-Seizure-Detection/
│
├── data/
│   ├── features.csv
│   ├── window_metadata.csv
│   └── multi_patient/
│       ├── chb02/
│       └── chb03/
│
├── images/
│   ├── cross-patient evaluation figures
│   ├── confusion matrices
│   ├── ROC / PR curves
│   └── SVG research tables and comparison figures
│
├── models/
│   ├── random_forest_model.pkl
│   └── phase2_class_weighted_recording_wise.pkl
│
├── notebooks/
│   └── 22 research notebooks
│
├── results/
│   ├── cross-patient result CSV files
│   ├── experiment metadata
│   ├── frozen experiment markers
│   └── research summaries
│
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── prediction.py
│   └── preprocessing.py
│
├── app.py
├── .gitignore
├── README.md
└── requirements.txt

Raw EDF recordings are excluded from version control.

23. Technologies Used

Python

MNE-Python

NumPy

Pandas

SciPy

Matplotlib

Scikit-learn

Joblib

Streamlit

Jupyter Notebook

24. Installation

Clone the repository

git clone https://github.com/ShivamKumarPrajapati-123/EEG-Seizure-Detection.git
cd EEG-Seizure-Detection

Install dependencies

pip install -r requirements.txt

25. Running the Project

Run the Streamlit application

streamlit run app.py

Notebook workflow

The notebooks can be followed progressively:

Reading EEG
    ↓
Understanding EEG
    ↓
Preprocessing
    ↓
Windowing
    ↓
Labeling
    ↓
Feature Extraction
    ↓
Model Training
    ↓
Dataset Construction
    ↓
Model Evaluation
    ↓
Prediction
    ↓
Recording-Wise Evaluation
    ↓
Event-Level Evaluation
    ↓
Cross-Validation
    ↓
Threshold Analysis
    ↓
Leakage-Free / Final Experiments
    ↓
Multi-Patient Dataset Construction
    ↓
CHB02 Cross-Patient Test
    ↓
CHB03 Cross-Patient Test

26. Reproducibility

The project uses fixed random seeds where applicable.

The final cross-patient experiment configuration is recorded in:

results/chb01_cross_patient_experiment_metadata.json

The experiment freeze markers record the final status for each
cross-patient evaluation:

results/chb02_experiment_frozen.txt
results/chb03_experiment_frozen.txt

The final frozen configuration uses:

Training patient = CHB01
Threshold = 0.05
Threshold source = CHB01 internal validation
CHB02 = final evaluation only
CHB03 = final evaluation only

No CHB02 or CHB03 threshold tuning was used for the final reported
cross-patient results.

27. Key Findings

The current experiments show that:

EEG seizure detection can be implemented as an end-to-end
signal-processing and machine-learning pipeline.

Severe class imbalance makes accuracy alone insufficient for
interpreting seizure detection.

Recording-wise evaluation provides a more realistic test of
unseen-recording generalization than random window-level splitting.

Threshold selection strongly affects sensitivity, precision, and
false-positive behavior.

The final frozen CHB01-trained model achieved measurable seizure
detection performance on both CHB02 and CHB03.

Performance varied substantially between the two unseen patients.

CHB02 achieved 66.67% sensitivity and 66.28% time-weighted seizure
coverage.

CHB03 achieved 25.47% sensitivity and 26.37% time-weighted seizure
coverage.

ROC-AUC values were higher than the corresponding precision-oriented
metrics, illustrating why multiple evaluation metrics are necessary
under severe class imbalance.

Event-level detection and window-level detection can provide
different views of system behavior.

Temporal consistency may help reduce isolated false-positive
detections, but the available temporal evaluation set is too small
for strong conclusions.

Observed feature-distribution differences between patients provide
evidence of inter-patient variability in the current feature space.

Patient-independent generalization has not yet been established.

28. Limitations

Severe class imbalance

The original CHB01 robustness dataset contains 13,080 normal windows and
only 101 seizure windows.

Limited number of patients

The final cross-patient study uses CHB01 for training and CHB02/CHB03
for testing. This is not sufficient to establish broad
patient-independent generalization.

Limited seizure-event sample

CHB02 contains three annotated seizure events and CHB03 contains seven
annotated seizure events in the evaluated recordings. Event-level
conclusions should therefore be interpreted cautiously.

Low precision at the frozen threshold

The final threshold of 0.05 increases sensitivity relative to more
conservative operating points but produces substantial false-positive
activity.

Threshold-selection limitation

Although the final threshold was selected using CHB01 internal
validation and not from CHB02/CHB03, a nested cross-validation procedure
would provide a stronger estimate of threshold-selection robustness.

Limited feature representation

The current feature set contains basic statistical and Welch PSD
features. More advanced time-frequency, nonlinear, spatial, and
channel-aware representations could be investigated.

No clinical validation

The system has not been validated in a clinical setting and should not
be used for diagnosis or treatment decisions.

29. Future Work

Future research may include:

Larger multi-patient evaluation.

Patient-independent cross-validation.

Nested threshold optimization.

Larger independent test cohorts.

Additional EEG channels and spatial features.

Time-frequency representations such as wavelets.

Nonlinear EEG features.

More robust temporal post-processing.

Improved false-positive suppression.

Support Vector Machines.

Gradient-boosting models.

XGBoost.

1D CNN architectures.

LSTM and other sequence models.

Transformer-based EEG models.

Patient-specific versus patient-independent modeling comparisons.

Seizure-type-specific analysis.

External-dataset validation.

Larger event-level evaluation.

Statistical confidence intervals and significance analysis.

More rigorous probability calibration assessment.

30. Research Status

Status: Research and Development / Learning Prototype

The project has progressed from basic EEG exploration to a frozen
multi-patient evaluation.

The final cross-patient experiment is:

CHB01 → CHB02
CHB01 → CHB03

with:

4-second windows
256 Hz sampling
8 features
Random Forest
200 estimators
class_weight="balanced"
threshold=0.05

The threshold was selected using CHB01 internal validation and frozen
before evaluation on CHB02 and CHB03.

The cross-patient experiments are complete and frozen.

31. Final Research Conclusion

This project developed and evaluated a machine-learning-based EEG
seizure detection pipeline using the CHB-MIT Scalp EEG Database.

The complete workflow includes EEG loading, signal preprocessing,
windowing, seizure labeling, statistical and spectral feature
extraction, Random Forest classification, class-imbalance handling,
recording-wise evaluation, threshold analysis, event-level evaluation,
temporal analysis, and cross-patient testing.

The recording-wise experiments demonstrated that high accuracy can
coexist with limited seizure sensitivity when seizure windows are rare.

The final frozen cross-patient experiments provide a more demanding test
by training on CHB01 and evaluating without retraining on CHB02 and
CHB03.

Using the frozen threshold of 0.05:

CHB02:
Sensitivity = 66.67%
Specificity = 94.53%
Precision   = 7.79%
F1-score    = 13.95%
ROC-AUC     = 93.08%
Coverage    = 66.28%
Delay       = 3.33 sec

and:

CHB03:
Sensitivity = 25.47%
Specificity = 98.50%
Precision   = 5.03%
F1-score    = 8.40%
ROC-AUC     = 86.58%
Coverage    = 26.37%
Delay       = 9.67 sec

These results demonstrate that performance can vary substantially when a
model trained on one patient is applied to other patients.

The lower false-positive event rate observed for CHB03 was accompanied
by lower seizure sensitivity and coverage, so it should not be
interpreted as an overall improvement.

The experiments therefore highlight the importance of patient diversity,
recording-wise validation, threshold discipline, event-level evaluation,
temporal analysis, and multiple performance metrics in EEG
seizure-detection research.

The current system remains a research and learning prototype. The
results do not establish clinical effectiveness or broad
patient-independent generalization.

Future work should focus on larger multi-patient cohorts, nested
threshold selection, improved feature representations, robust temporal
modeling, false-positive analysis, and independent external validation.

32. Disclaimer

This project is developed for educational and research purposes only.

The model is not a clinically validated medical device and must not
be used for:

Medical diagnosis

Treatment decisions

Clinical decision-making

Reported results are based on a limited number of patients, recordings,
and seizure events. They should not be interpreted as evidence of
clinical effectiveness or broad patient-independent generalization.

33. Author

Shivam Prajapati

Computer Science Engineering --- Artificial Intelligence & Machine
Learning

GitHub:
https://github.com/ShivamKumarPrajapati-123/EEG-Seizure-Detection