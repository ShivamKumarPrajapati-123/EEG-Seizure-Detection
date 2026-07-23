# EEG-Based Epileptic Seizure Detection Using Signal Processing and Machine Learning

## Author

**Shivam Prajapati**  
Computer Science Engineering — Artificial Intelligence & Machine Learning

---

## 1. Project Overview

Epileptic seizures are neurological events that can produce abnormal patterns in Electroencephalography (EEG) signals. Automatic seizure detection from EEG recordings is an important application of biomedical signal processing and machine learning.

This project presents an end-to-end machine learning pipeline for detecting epileptic seizure activity from EEG signals using the **CHB-MIT Scalp EEG Database**.

The system processes raw EEG recordings, performs signal preprocessing, segments continuous EEG signals into fixed-length windows, assigns seizure and non-seizure labels, extracts statistical and frequency-domain features, and applies a Random Forest classifier for binary classification.

The project also evaluates the effect of severe class imbalance and investigates classification threshold adjustment to understand the trade-off between seizure sensitivity and false-positive predictions.

The complete pipeline includes:

- EEG data loading from EDF files
- EEG signal preprocessing
- Fixed-length EEG windowing
- Seizure and non-seizure labeling
- Statistical feature extraction
- Frequency-domain feature extraction
- Machine learning dataset construction
- Random Forest model training
- Model evaluation
- Classification threshold analysis
- EEG window-level prediction
- Automated evaluation result generation
- Result visualization and reporting

This project was developed as a research and learning prototype to gain practical experience in:

- Biomedical signal processing
- EEG data analysis
- Feature engineering
- Machine learning
- Imbalanced classification
- Model evaluation
- Reproducible machine learning workflows

---

## 2. Objectives

The main objectives of this project are:

1. Load EEG recordings from EDF files.
2. Understand EEG signal properties and metadata.
3. Preprocess EEG signals using frequency filtering.
4. Segment continuous EEG recordings into fixed-length windows.
5. Identify seizure and non-seizure EEG windows.
6. Extract statistical features from EEG signals.
7. Extract frequency-domain features using Power Spectral Density (PSD).
8. Construct a machine learning feature dataset.
9. Train a Random Forest classifier.
10. Evaluate model performance using appropriate classification metrics.
11. Analyze the effect of severe class imbalance.
12. Investigate classification threshold adjustment.
13. Perform prediction on individual EEG windows.
14. Generate reproducible evaluation results and visualizations.

---

## 3. Dataset

This project uses the **CHB-MIT Scalp EEG Database**, a publicly available EEG dataset containing EEG recordings from pediatric subjects with intractable seizures.

The EEG recordings are provided in **EDF (European Data Format)** files and are processed using the **MNE-Python** library.

The current project focuses on EEG recordings from the CHB-MIT dataset, including analysis of:

```text
chb01_03.edf