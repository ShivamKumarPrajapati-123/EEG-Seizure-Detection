# 🧠 EEG-Based Epileptic Seizure Detection Using Signal Processing and Machine Learning

**Author:** Shivam Prajapati  
**Program:** Computer Science Engineering — Artificial Intelligence & Machine Learning

---

## 1. Project Overview

Epileptic seizures are neurological events that can produce abnormal patterns in electroencephalography (EEG) signals. Automatic seizure detection from EEG recordings is an important application of biomedical signal processing and machine learning.

This project presents an end-to-end research and learning pipeline for analyzing EEG recordings and detecting seizure-related activity using signal processing, feature engineering, and machine learning.

The project uses the **CHB-MIT Scalp EEG Database** and processes EEG recordings stored in EDF (European Data Format) files.

The overall pipeline includes:

- EEG data loading from EDF files
- EEG signal preprocessing
- Fixed-length EEG windowing
- Seizure and non-seizure labeling
- Statistical feature extraction
- Frequency-domain feature extraction using Power Spectral Density (PSD)
- Random Forest machine learning classification
- Class-imbalance handling
- Recording-wise evaluation
- Classification threshold analysis
- Recording-wise 5-fold cross-validation
- Window-level seizure detection
- Recording-level seizure detection
- Seizure event-level evaluation
- Temporal consecutive-window analysis
- Integrated final evaluation
- Streamlit-based interactive deployment
- Final research interpretation and conclusion

The project was developed as a **research and learning prototype** to gain practical experience in:

- Biomedical signal processing
- EEG data analysis
- Feature engineering
- Machine learning
- Imbalanced classification
- Model evaluation
- Cross-validation
- Event-level evaluation
- Temporal post-processing
- Reproducible research workflows
- Machine learning application deployment

> **Important:** This project is intended for educational and research purposes. It is **not a clinically validated medical diagnostic system** and must not be used for medical diagnosis or clinical decision-making.

---

## 2. Objectives

The main objectives of this project are:

- Load EEG recordings from EDF files.
- Understand EEG signal properties and metadata.
- Preprocess EEG signals using frequency filtering.
- Segment continuous EEG recordings into fixed-length windows.
- Identify seizure and non-seizure EEG windows.
- Extract statistical features from EEG signals.
- Extract frequency-domain features using Power Spectral Density (PSD).
- Construct a machine learning feature dataset.
- Train a Random Forest classifier.
- Address severe class imbalance using class-weighted learning.
- Evaluate model performance using appropriate classification metrics.
- Analyze the effect of classification threshold selection.
- Evaluate generalization using recording-wise cross-validation.
- Evaluate seizure detection at the event level.
- Investigate temporal consecutive-window post-processing.
- Generate reproducible evaluation results and visualizations.
- Deploy the prediction pipeline as an interactive Streamlit application.
- Identify limitations and future research directions.

---

## 3. Dataset

This project uses the **CHB-MIT Scalp EEG Database**, a publicly available EEG dataset containing long-term scalp EEG recordings from pediatric subjects with intractable seizures.

The EEG recordings are provided in **EDF (European Data Format)** files and are processed using the **MNE-Python** library.

The project includes analysis of multiple EEG recordings from the CHB-MIT dataset.

The raw EDF recordings are not included in the GitHub repository because of their large file size. They should be obtained separately from the CHB-MIT Scalp EEG Database and supplied to the local processing pipeline.

### Final Robustness Evaluation Dataset

The final recording-wise cross-validation experiment used:

| Parameter | Value |
|---|---:|
| Total EEG recordings | 15 |
| Seizure-containing recordings | 6 |
| Normal-only recordings | 9 |
| Total EEG windows | 13,181 |
| Normal windows | 13,080 |
| Seizure windows | 101 |
| Number of features | 8 |

The final robustness dataset is highly imbalanced, with seizure windows representing a small proportion of the total windows.

This class imbalance is an important consideration when interpreting model performance.

---

## 4. Complete System Workflow

The overall pipeline is:

```text
Raw EEG EDF Recordings
        │
        ▼
EEG Data Loading using MNE
        │
        ▼
EEG Metadata and Signal Inspection
        │
        ▼
Signal Preprocessing
        │
        ▼
4-Second EEG Windowing
        │
        ▼
Seizure / Non-Seizure Labeling
        │
        ▼
Feature Extraction
        │
        ├──────────────────────────────┐
        ▼                              ▼
Statistical Features         Frequency-Domain Features
        │                              │
        └──────────────┬───────────────┘
                       ▼
             Machine Learning Dataset
                       │
                       ▼
             Class-Weighted Random Forest
                       │
                       ▼
            Probability-Based Predictions
                       │
                       ├──────────────────────────┐
                       ▼                          ▼
              Threshold Analysis        Recording-Wise
                                         Cross-Validation
                       │                          │
                       └──────────────┬───────────┘
                                      ▼
                           Integrated Evaluation
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
                 Event-Level              Temporal Consecutive-
                  Evaluation                Window Analysis
                       │                             │
                       └──────────────┬──────────────┘
                                      ▼
                           Final Research Conclusion
                                      │
                                      ▼
                              Streamlit Deployment
```

---

## 5. EEG Signal Preprocessing

The raw EEG signals are loaded from EDF files using MNE-Python.

A band-pass filter is applied to retain relevant EEG frequency components and reduce unwanted low-frequency drift and high-frequency noise.

The filtering range used in the current pipeline is:

| Parameter | Value |
|---|---:|
| Low Cutoff Frequency | 0.5 Hz |
| High Cutoff Frequency | 40 Hz |

The preprocessing stage prepares EEG signals for subsequent windowing and feature extraction.

---

## 6. EEG Windowing

The continuous EEG recording is divided into fixed-length windows.

The prediction pipeline uses:

| Parameter | Value |
|---|---:|
| Sampling Frequency | 256 Hz |
| Window Duration | 4 seconds |
| Samples per Window | 1024 |

For recordings containing 23 EEG channels, each window can be represented as:

```text
23 EEG Channels × 1024 Samples
```

For example, a 3600-second recording sampled at 256 Hz produces:

```text
900 complete 4-second EEG windows
```

The fixed-length windows are treated as individual samples for feature extraction and machine-learning prediction.

---

## 7. Seizure Labeling

During dataset construction, each EEG window was assigned a binary label:

```text
0 → Normal / Non-Seizure
1 → Seizure
```

Seizure annotations associated with the EEG recordings were used to identify windows overlapping with seizure activity.

The final robustness evaluation contained:

```text
Total Windows   = 13,181
Normal Windows  = 13,080
Seizure Windows = 101
```

This produced a severe class imbalance between normal and seizure samples.

> **Important:** Earlier experiments used smaller datasets and smaller test sets. Those experiments are retained as part of the project's experimental progression, while the final robustness evaluation is based on the larger 15-recording dataset described above.

Because of the class imbalance, overall accuracy alone is not considered sufficient for evaluating seizure detection performance.

---

## 8. Feature Extraction

Features are extracted from each EEG window using statistical and frequency-domain analysis.

The final feature representation contains eight features.

### 8.1 Statistical Features

The following statistical features are extracted:

- Mean
- Standard Deviation
- Variance

These features provide statistical representations of EEG signal characteristics.

### 8.2 Frequency-Domain Features

Power Spectral Density (PSD) is calculated using the Welch method.

The following EEG frequency bands are analyzed:

| Frequency Band | Frequency Range |
|---|---:|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–40 Hz |

The final feature vector contains:

```text
Mean
Std
Variance
Delta
Theta
Alpha
Beta
Gamma
```

These features combine statistical and spectral information from the EEG signal.

---

## 9. Machine Learning Model

A Random Forest Classifier is used as the primary machine-learning model.

Random Forest was selected because it:

- Works effectively with tabular feature data.
- Can model nonlinear relationships.
- Does not require extensive feature scaling.
- Provides feature importance information.
- Is relatively straightforward to train and interpret.
- Provides probability estimates that can be used for threshold analysis.

The model uses class weighting to reduce the effect of severe class imbalance.

The trained model is stored as:

```text
models/random_forest_model.pkl
```

The Random Forest implementation is considered a research baseline model and is not a clinically validated seizure detection model.

---

## 10. Model Evaluation

The project evaluates model performance at multiple levels.

### 10.1 Window-Level Evaluation

Individual EEG windows are classified as:

- Normal / Non-Seizure
- Seizure

The model is evaluated using:

- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- F1-Score
- Balanced Accuracy
- Confusion Matrix
- Classification Report

Because the dataset is severely imbalanced, accuracy alone is not considered sufficient.

### 10.2 Recording-Level Evaluation

Recording-wise evaluation is used to determine whether the model can detect seizure-related activity in EEG recordings that were not used for model training.

Recording-wise evaluation is more realistic than random window-level splitting because windows from the same EEG recording can be highly correlated.

### 10.3 Recording-Wise 5-Fold Cross-Validation

The final robustness evaluation used:

| Parameter | Value |
|---|---:|
| Cross-validation method | StratifiedGroupKFold |
| Number of folds | 5 |
| Random state | 42 |
| Grouping variable | EEG recording filename |

The grouping strategy ensures that windows from the same EEG recording are not simultaneously present in the training and testing sets within a fold.

The final experiment used:

```text
Total EEG recordings          = 15
Seizure-containing recordings = 6
Normal-only recordings        = 9
Total EEG windows             = 13,181
```

Recording leakage checks confirmed that recordings were separated between training and testing within each fold.

This provides a more realistic estimate of generalization to unseen EEG recordings than random window-level splitting.

However, because the dataset contains a limited number of patients, this evaluation does not establish generalization to completely unseen patients.

---

## 11. Experimental Progression

The project was developed through multiple experimental stages.

### Experiment 03 — Recording-Wise Holdout

The recording-wise holdout evaluation established a stronger evaluation protocol than random window-level splitting.

At the selected threshold of 0.50, the experiment achieved:

| Metric | Result |
|---|---:|
| Accuracy | 98.73% |
| Precision | 83.33% |
| Sensitivity | 34.25% |
| Specificity | 99.88% |
| F1-Score | 48.54% |
| Balanced Accuracy | 67.06% |

This experiment demonstrated very high specificity but limited seizure sensitivity.

### Experiment 04 — Threshold Optimization

Experiment 04 investigated the effect of changing the classification threshold.

The selected operating point was:

```text
Threshold = 0.10
```

The recording-wise holdout result was:

| Metric | Result |
|---|---:|
| Accuracy | 98.97% |
| Precision | 65.62% |
| Sensitivity | 86.30% |
| Specificity | 99.20% |
| F1-Score | 74.56% |
| Balanced Accuracy | 92.75% |

The experiment demonstrated that lowering the threshold increased seizure sensitivity while increasing false-positive predictions.

However, this threshold analysis was exploratory and was later subjected to a stronger recording-wise cross-validation evaluation.

### Experiment 05 — Recording/Event-Level Holdout

Experiment 05 evaluated the model at a recording/event level.

The evaluation produced:

| Metric | Result |
|---|---:|
| Accuracy | 60.00% |
| Precision | 60.00% |
| Sensitivity | 100.00% |
| Specificity | 0.00% |
| F1-Score | 75.00% |

This experiment demonstrated the importance of evaluating seizure detection at a higher level than individual windows.

However, the evaluation set was very small and produced a large number of false-positive recording detections.

Therefore, these results are considered exploratory and are not used as evidence of clinical generalization.

### Experiment 06 — Recording-Wise 5-Fold Cross-Validation

Experiment 06 provided the strongest robustness evaluation currently available in the project.

The evaluation used recording-wise 5-fold cross-validation, ensuring that recordings were separated between training and testing.

Two probability thresholds were evaluated using out-of-fold predictions:

```text
Threshold = 0.50
Threshold = 0.10
```

The results are presented below.

---

## 12. Final Cross-Validation Results

### Threshold 0.50

The conventional threshold of 0.50 produced:

| Metric | Result |
|---|---:|
| Accuracy | 99.48% |
| Precision | 76.19% |
| Sensitivity | 47.52% |
| Specificity | 99.89% |
| F1-Score | 58.54% |
| Balanced Accuracy | 73.71% |

#### Pooled Confusion Matrix

```text
True Negatives  = 13,065
False Positives = 15
False Negatives = 53
True Positives  = 48
```

This operating point provides very high specificity and relatively high precision, but detects fewer seizure windows.

### Threshold 0.10

The sensitivity-oriented threshold of 0.10 produced:

| Metric | Result |
|---|---:|
| Accuracy | 99.27% |
| Precision | 51.75% |
| Sensitivity | 73.27% |
| Specificity | 99.47% |
| F1-Score | 60.66% |
| Balanced Accuracy | 86.37% |

#### Pooled Confusion Matrix

```text
True Negatives  = 13,011
False Positives = 69
False Negatives = 27
True Positives  = 74
```

### Comparison

Compared with threshold 0.50:

**Sensitivity**

```text
47.52% → 73.27%
```

Increase:

```text
25.75 percentage points
```

**Balanced Accuracy**

```text
73.71% → 86.37%
```

Increase:

```text
12.66 percentage points
```

However:

**Precision**

```text
76.19% → 51.75%
```

Decrease:

```text
24.44 percentage points
```

The number of false-positive predictions increased from:

```text
15 → 69
```

while false-negative predictions decreased from:

```text
53 → 27
```

### Interpretation

The results demonstrate an important sensitivity-specificity trade-off.

The threshold of 0.50 provides a more conservative operating point with higher precision and specificity.

The threshold of 0.10 provides a more sensitivity-oriented operating point and detects more seizure windows, but at the cost of additional false-positive predictions.

For a seizure-detection research prototype, threshold 0.10 may be preferable when missing a seizure is considered more costly than generating additional false-positive predictions.

However, the appropriate operating threshold depends on the intended application and should be selected using a rigorous validation strategy.

---

## 13. Threshold Optimization Limitation

The threshold of 0.10 was originally identified during the exploratory threshold analysis in Experiment 04.

It was subsequently evaluated using out-of-fold predictions generated during the recording-wise cross-validation experiment.

Therefore, the threshold comparison provides useful evidence about the behavior of the model on unseen recordings, but it should not be interpreted as a fully nested threshold-optimization experiment.

A more rigorous evaluation should:

1. Split recordings into training and validation/test sets.
2. Select the optimal threshold using only the training/validation portion.
3. Keep the final test recordings completely untouched.
4. Evaluate the selected threshold only on the held-out test recordings.
5. Repeat the process within each cross-validation fold if cross-validation is used.

This nested threshold-selection strategy is an important direction for future work.

---

## 14. Event-Level Seizure Detection

The final integrated evaluation also examined seizure detection at the event level.

Using the sensitivity-oriented threshold:

```text
Threshold = 0.10
```

the evaluation included:

```text
Seizure-containing recordings = 3
Detected seizure events        = 3
Missed seizure events          = 0
Event-level sensitivity        = 100.00%
```

### Interpretation

The model successfully detected the seizure event in all three seizure-containing recordings included in this evaluation.

However, this result must be interpreted carefully.

The event-level evaluation was performed on a small evaluation set containing only:

```text
3 seizure-containing recordings
2 normal-only recordings
```

Therefore, the observed 100% event-level sensitivity is considered exploratory and should not be interpreted as evidence of clinical generalization.

A larger multi-patient event-level evaluation is required to establish reliable seizure-event detection performance.

---

## 15. Temporal Consecutive-Window Analysis

A temporal post-processing strategy was evaluated by requiring multiple consecutive positive EEG windows before declaring a recording-level seizure detection.

Each window represents:

```text
4 seconds
```

The results were:

| Consecutive Positive Windows | Time Requirement | Accuracy | Precision | Sensitivity | Specificity | F1-Score |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4 sec | 60% | 60% | 100% | 0% | 75% |
| 2 | 8 sec | 80% | 75% | 100% | 50% | 85.71% |
| 3 | 12 sec | 80% | 75% | 100% | 50% | 85.71% |
| 5 | 20 sec | 100% | 100% | 100% | 100% | 100% |

The results suggest that requiring multiple consecutive positive windows can reduce false-positive recording detections while maintaining seizure-event sensitivity.

At:

```text
1 consecutive positive window
```

the evaluation produced:

```text
Sensitivity = 100%
Specificity = 0%
```

At:

```text
5 consecutive positive windows
20 seconds
```

the evaluation produced:

```text
Sensitivity = 100%
Specificity = 100%
```

### Important Limitation

The temporal analysis was performed on a very small evaluation set containing only:

```text
3 seizure-containing recordings
2 normal-only recordings
```

Therefore, the observed 100% sensitivity and 100% specificity at the 20-second configuration should be considered exploratory.

These results should not be interpreted as evidence of clinical generalization.

The analysis demonstrates the potential value of temporal consistency, but the strategy requires validation on a much larger multi-patient dataset.

---

## 16. Final Integrated Evaluation

The final integrated evaluation was performed in:

```text
14_Final_Integrated_Evaluation.ipynb
```

The notebook consolidated:

- Recording-wise evaluation
- Threshold optimization
- Recording-wise 5-fold cross-validation
- Cross-validation threshold comparison
- Seizure event-level evaluation
- Temporal consecutive-window analysis
- Integrated experiment comparison
- Consistency verification
- Research interpretation
- Final research conclusion

The final evaluation confirmed that the results were internally consistent with the previously generated Experiment 04, Experiment 05, and Experiment 06 outputs.

The final integrated outputs include:

```text
results/final_integrated_research_conclusion.txt
results/final_integrated_metrics.csv
results/final_temporal_consecutive_window_results.csv
results/final_event_level_summary.csv
```

---

## 17. Key Findings

The main findings of the current project are:

- The project successfully implements an end-to-end EEG seizure detection pipeline.
- The baseline model demonstrated that high accuracy alone can be misleading under severe class imbalance.
- Recording-wise evaluation provides a more realistic estimate of generalization to unseen EEG recordings than random window-level splitting.
- Recording-wise 5-fold cross-validation provided the strongest robustness evidence currently available in the project.
- The Random Forest model maintained very high specificity across unseen recordings.
- Seizure sensitivity varied between cross-validation folds, indicating recording-to-recording variability.
- Reducing the classification threshold from 0.50 to 0.10 increased pooled seizure sensitivity from 47.52% to 73.27%.
- Balanced accuracy increased from 73.71% to 86.37%.
- The sensitivity improvement was accompanied by a reduction in precision from 76.19% to 51.75%.
- False-positive predictions increased when using the lower threshold.
- Event-level analysis detected all three seizure events in the small evaluation set.
- Temporal consecutive-window analysis suggested that requiring multiple positive windows may reduce false-positive recording detections.
- The event-level and temporal results are exploratory because they were obtained from small evaluation sets.
- The current evaluation does not establish patient-independent generalization.
- The system remains a research and learning prototype rather than a clinically validated diagnostic system.

---

## 18. Limitations

### 18.1 Severe Class Imbalance

The final robustness dataset contains:

```text
Normal Windows  = 13,080
Seizure Windows = 101
```

The large difference between the two classes makes accuracy alone an inadequate measure of model performance.

### 18.2 Limited Number of Recordings

The final robustness evaluation uses:

```text
15 EEG recordings
```

Only:

```text
6 recordings contain seizure activity
```

This limits the statistical strength of the conclusions.

### 18.3 Limited Patient Diversity

Recording-wise cross-validation evaluates generalization to unseen recordings.

However, the current evaluation does not establish generalization to completely unseen patients.

Patient-independent evaluation using a larger multi-patient dataset is required.

### 18.4 Recording-to-Recording Variability

Seizure sensitivity varied across cross-validation folds.

This indicates that EEG characteristics and seizure patterns can vary substantially between recordings.

This variability may affect model performance when applied to new EEG recordings.

### 18.5 Threshold Optimization Limitation

The threshold of 0.10 was originally identified during Experiment 04 and later evaluated using Experiment 06 out-of-fold predictions.

Therefore, the current evaluation is not a fully nested threshold-optimization framework.

Future work should perform threshold selection independently within each training fold.

### 18.6 Small Event-Level Evaluation Set

The final event-level evaluation included only:

```text
3 seizure-containing recordings
2 normal-only recordings
```

Therefore, the observed 100% event-level sensitivity should be considered exploratory.

### 18.7 Small Temporal Evaluation Set

The consecutive-window analysis was also performed on a small evaluation set.

The observed 100% sensitivity and 100% specificity at the 20-second configuration may not generalize to larger datasets.

### 18.8 Limited Feature Representation

The current model uses a relatively small feature set consisting primarily of:

- Statistical features
- Frequency-band features

More advanced time-frequency, nonlinear, and spatial EEG features may improve the representation of complex seizure patterns.

### 18.9 Limited Model Complexity

The primary model is a Random Forest classifier.

More advanced approaches may provide improved performance but require careful validation to avoid overfitting.

### 18.10 No Clinical Validation

This project is an educational and research prototype.

The model has not been clinically validated and should not be used for:

- Medical diagnosis
- Treatment decisions
- Clinical decision-making

---

## 19. Future Work

Future improvements may include:

- Using a larger number of EEG recordings.
- Including data from multiple patients.
- Performing patient-independent evaluation.
- Increasing the number of seizure samples.
- Performing nested threshold optimization.
- Selecting thresholds using dedicated validation sets.
- Applying improved class imbalance handling techniques.
- Investigating resampling strategies.
- Evaluating Precision-Recall curves.
- Evaluating ROC-AUC.
- Testing Support Vector Machines (SVM).
- Testing XGBoost and other ensemble methods.
- Exploring 1D CNN-based EEG classification.
- Exploring LSTM and other deep learning approaches.
- Extracting additional time-domain features.
- Extracting additional frequency-domain features.
- Investigating time-frequency representations such as wavelets.
- Exploring nonlinear EEG features.
- Investigating channel-selection strategies.
- Evaluating performance across multiple patients.
- Evaluating performance across different seizure types.
- Improving seizure detection sensitivity while controlling false-positive rates.
- Investigating robust temporal post-processing strategies.
- Performing larger-scale event-level evaluation.
- Comparing patient-specific and patient-independent seizure detection models.
- Validating the final system on an independent external dataset.

---

## 20. Technologies Used

The project uses:

- Python
- MNE-Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 21. Project Structure

A representative project structure is:

```text
EEG-Seizure-Detection/
│
├── data/
│   ├── features.csv
│   ├── features.npy
│   ├── labels.npy
│   └── README.md
│
├── images/
│   ├── random_forest_baseline_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   └── random_forest_threshold_confusion_matrix.png
│
├── models/
│   └── random_forest_model.pkl
│
├── notebooks/
│   ├── 01_Reading_EEG.ipynb
│   ├── 02_Understanding_EEG.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Windowing.ipynb
│   ├── 05_Labeling.ipynb
│   ├── 06_Feature_Extraction.ipynb
│   ├── 07_Model_Training.ipynb
│   ├── 08_Dataset_Builder.ipynb
│   ├── 09_Model_Evaluation.ipynb
│   ├── 10_Prediction.ipynb
│   ├── 13_Robustness_Cross_Validation.ipynb
│   └── 14_Final_Integrated_Evaluation.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── final_evaluation.txt
│   ├── threshold_analysis.png
│   ├── experiment_04_threshold_comparison.csv
│   ├── experiment_04_threshold_optimization.txt
│   ├── experiment_comparison_summary.csv
│   ├── experiment_06_cross_validation_report.txt
│   ├── final_integrated_research_conclusion.txt
│   ├── final_integrated_metrics.csv
│   ├── final_temporal_consecutive_window_results.csv
│   └── final_event_level_summary.csv
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
```

Raw EDF recordings and temporary uploaded files are excluded from version control.

The exact contents of the repository may evolve as additional experiments and result artifacts are added.

---

## 22. Installation

### Clone the Repository

```bash
git clone https://github.com/ShivamKumarPrajapati-123/EEG-Seizure-Detection.git
```

### Navigate to the Project Directory

```bash
cd EEG-Seizure-Detection
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 23. Running the Project Locally

### Run the Streamlit Application

From the project root:

```bash
streamlit run app.py
```

The application will open in your browser.

### Run the Notebook Workflow

The notebooks demonstrate the individual stages of the project pipeline.

The general workflow is:

```text
01_Reading_EEG
        ↓
02_Understanding_EEG
        ↓
03_Preprocessing
        ↓
04_Windowing
        ↓
05_Labeling
        ↓
06_Feature_Extraction
        ↓
08_Dataset_Builder
        ↓
07_Model_Training
        ↓
09_Model_Evaluation
        ↓
10_Prediction
        ↓
13_Robustness_Cross_Validation
        ↓
14_Final_Integrated_Evaluation
```

The later notebooks focus on robustness, cross-validation, integrated evaluation, event-level analysis, and final research interpretation.

---

## 24. Reproducibility

The project uses fixed random seeds where applicable to improve reproducibility of machine-learning experiments.

Important processed artifacts include:

```text
data/features.csv
data/features.npy
data/labels.npy
models/random_forest_model.pkl
```

Final integrated evaluation outputs include:

```text
results/final_integrated_research_conclusion.txt
results/final_integrated_metrics.csv
results/final_temporal_consecutive_window_results.csv
results/final_event_level_summary.csv
```

Raw EEG recordings are not stored in the repository and must be obtained separately from the CHB-MIT Scalp EEG Database.

For rigorous reproduction of the final results, the experiment notebooks and corresponding result files should be executed and reviewed in the documented experimental sequence.

---

## 25. Research Status

**Status:** Research and Development / Learning Prototype

The project demonstrates an end-to-end EEG seizure detection pipeline covering:

- EEG data loading
- Signal preprocessing
- Window segmentation
- Seizure labeling
- Feature extraction
- Dataset construction
- Machine-learning model training
- Recording-wise evaluation
- Threshold analysis
- Recording-wise cross-validation
- Event-level evaluation
- Temporal consecutive-window analysis
- Integrated research evaluation
- Streamlit deployment

The final evaluation demonstrates that:

- Recording-wise cross-validation provides stronger evidence of generalization to unseen EEG recordings than random window-level splitting.
- Threshold selection significantly affects seizure sensitivity and false-positive behavior.
- A lower threshold can improve sensitivity but reduce precision.
- Temporal consistency may help reduce false-positive recording detections.
- Patient-independent generalization remains unestablished.

The current system should therefore be considered a research and learning prototype rather than a clinically validated seizure detection system.

---

## 26. Final Research Conclusion

This project developed and evaluated a machine-learning-based EEG seizure detection pipeline using the CHB-MIT Scalp EEG dataset.

The complete pipeline included EEG signal preprocessing, window-based segmentation, seizure/non-seizure labeling, feature extraction, machine-learning classification, threshold optimization, recording-wise evaluation, seizure event-level evaluation, recording-wise cross-validation, and temporal consistency analysis.

The initial experiments demonstrated that conventional accuracy alone is insufficient for assessing seizure detection performance because of the strong class imbalance between seizure and non-seizure EEG windows.

The final recording-wise 5-fold cross-validation provided the strongest robustness evidence currently available in this project.

At the default threshold of 0.50, the model achieved:

```text
Accuracy          = 99.48%
Precision         = 76.19%
Sensitivity       = 47.52%
Specificity       = 99.89%
F1-Score          = 58.54%
Balanced Accuracy = 73.71%
```

At the sensitivity-oriented threshold of 0.10:

```text
Accuracy          = 99.27%
Precision         = 51.75%
Sensitivity       = 73.27%
Specificity       = 99.47%
F1-Score          = 60.66%
Balanced Accuracy = 86.37%
```

The reduction in threshold increased seizure sensitivity by 25.75 percentage points and improved balanced accuracy by 12.66 percentage points.

However, this improvement was accompanied by a reduction in precision and an increase in false-positive predictions.

These results demonstrate an important sensitivity-specificity trade-off in EEG seizure detection.

The event-level analysis detected all three seizure events in the evaluated seizure-containing recordings. The temporal consecutive-window analysis further suggested that requiring multiple consecutive positive windows may reduce false-positive recording detections while maintaining seizure-event sensitivity.

However, both event-level and temporal results were obtained from small evaluation sets and should therefore be considered exploratory.

The most important limitation of the current study is the limited patient diversity of the dataset. Although recording-wise cross-validation provides a stronger estimate of generalization to unseen recordings, it does not establish generalization to completely unseen patients.

Therefore, the current system should be considered a research and learning prototype rather than a clinically validated diagnostic tool.

Future work should include patient-independent validation using a larger multi-patient EEG dataset, rigorous nested threshold optimization, evaluation across different seizure types, improved handling of class imbalance, analysis of false-positive detections, optimization of temporal post-processing strategies, and comparison with more advanced machine-learning and deep-learning approaches.

Overall, the project demonstrates the feasibility of developing an EEG-based seizure detection system and highlights the importance of recording-wise validation, threshold analysis, event-level evaluation, temporal consistency analysis, and appropriate performance metrics when assessing seizure detection systems.

---

## 27. Disclaimer

This project is developed for educational and research purposes only.

The current model is not a clinically validated medical device and should not be used for:

- Medical diagnosis
- Treatment decisions
- Clinical decision-making

The reported results are based on limited recordings and patients, and some event-level and temporal analyses were performed on small evaluation sets.

The reported performance should therefore not be interpreted as evidence of clinical effectiveness or patient-independent generalization.

---

## 28. Author

**Shivam Prajapati**

Computer Science Engineering — Artificial Intelligence & Machine Learning

**GitHub:**  
https://github.com/ShivamKumarPrajapati-123