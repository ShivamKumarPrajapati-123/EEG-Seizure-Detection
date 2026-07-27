# 🧠 EEG-Based Epileptic Seizure Detection Using Signal Processing and Machine Learning

**Author:** Shivam Prajapati  
**Program:** Computer Science Engineering — Artificial Intelligence & Machine Learning

---

## 1. Project Overview

Epileptic seizures are neurological events that can produce abnormal patterns in electroencephalography (EEG) signals. Automatic seizure detection from EEG recordings is an important application of biomedical signal processing and machine learning.

This project presents an end-to-end research and learning pipeline for analyzing EEG recordings and detecting seizure-like activity using signal processing, feature engineering, and machine learning.

The project uses the **CHB-MIT Scalp EEG Database** and processes EEG recordings stored in EDF (European Data Format) files.

The overall pipeline includes:

- EEG data loading from EDF files
- EEG signal preprocessing
- Fixed-length EEG windowing
- Seizure and non-seizure labeling
- Statistical feature extraction
- Frequency-domain feature extraction using Power Spectral Density (PSD)
- Random Forest model training
- Model evaluation
- Classification threshold analysis
- Window-level prediction
- Event-level seizure-like activity detection
- Streamlit-based interactive deployment
- Result visualization and downloadable outputs

The project was developed as a **research and learning prototype** to gain practical experience in:

- Biomedical signal processing
- EEG data analysis
- Feature engineering
- Machine learning
- Imbalanced classification
- Model evaluation
- Reproducible workflows
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
- Evaluate model performance using appropriate classification metrics.
- Analyze the effect of severe class imbalance.
- Investigate classification threshold adjustment.
- Perform prediction on individual EEG windows.
- Generate reproducible evaluation results and visualizations.
- Deploy the prediction pipeline as an interactive Streamlit application.

---

## 3. Dataset

This project uses the **CHB-MIT Scalp EEG Database**, a publicly available EEG dataset containing EEG recordings from pediatric subjects with intractable seizures.

The EEG recordings are provided in **EDF (European Data Format)** files and are processed using the **MNE-Python** library.

The project includes analysis of recordings from the CHB-MIT dataset, including:

```text
chb01_03.edf
```

The raw EDF recordings are not included in the GitHub repository because of their large file size. They should be obtained separately from the CHB-MIT Scalp EEG Database and supplied to the application or local processing pipeline.

---

## 4. Complete System Workflow

The overall pipeline is:

```text
Raw EEG EDF Recording
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
             Random Forest Classifier
                       │
                       ▼
                 Model Evaluation
                       │
                       ▼
            Classification Threshold
                  Analysis
                       │
                       ▼
             Window-Level Prediction
                       │
                       ▼
          Event-Level Detection Timeline
                       │
                       ▼
              Streamlit Deployment
                       │
                       ▼
              Downloadable Results
```

---

## 5. EEG Signal Preprocessing

The raw EEG signals are loaded from EDF files using **MNE-Python**.

A band-pass filter is applied to retain relevant EEG frequency components and reduce unwanted low-frequency drift and high-frequency noise.

The filtering range used in the current pipeline is:

| Parameter | Value |
|---|---:|
| Low Cutoff Frequency | 0.5 Hz |
| High Cutoff Frequency | 40 Hz |

The preprocessing step prepares the EEG signals for subsequent windowing and feature extraction.

---

## 6. EEG Windowing

The continuous EEG recording is divided into fixed-length windows.

The current prediction pipeline uses:

| Parameter | Value |
|---|---:|
| Sampling Frequency | 256 Hz |
| Window Duration | 4 seconds |
| Samples per Window | 1024 |

Each EEG window contains:

```text
23 EEG Channels × 1024 Samples
```

For a 3600-second recording at 256 Hz, the deployed application created:

```text
900 complete 4-second EEG windows
```

The fixed-length windows are treated as individual samples for feature extraction and prediction.

---

## 7. Seizure Labeling

During the dataset construction experiments, each EEG window was assigned a binary label:

```text
0 → Normal / Non-Seizure
1 → Seizure
```

Seizure annotations associated with the EEG recordings were used to identify windows overlapping with seizure activity.

The experimental dataset used for model evaluation contained approximately:

```text
Total Windows   = 900
Normal Windows  = 890
Seizure Windows = 10
```

This produced a severe class imbalance between normal and seizure samples.

> **Note:** The counts above describe the experimental labeled dataset used for model evaluation. They should not be confused with the deployed application's prediction output, where the recording is classified into **Normal** and **Seizure-like** windows using a probability threshold.

---

## 8. Feature Extraction

Features are extracted from each EEG window using statistical and frequency-domain analysis.

The current feature dataset contains eight features.

### 8.1 Statistical Features

The following statistical features are extracted:

- Mean
- Standard Deviation
- Variance

These features provide statistical representations of the EEG signal.

### 8.2 Frequency-Domain Features

Power Spectral Density (PSD) is calculated using the **Welch method**.

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

The experimental feature dataset was represented as:

```text
Features Shape = (900, 8)
Labels Shape   = (900,)
```

---

## 9. Machine Learning Model

A **Random Forest Classifier** is used as the baseline machine learning model.

Random Forest was selected because it:

- Works effectively with tabular feature data.
- Can model nonlinear relationships.
- Does not require extensive feature scaling.
- Provides feature importance information.
- Is relatively straightforward to train and interpret.

The trained model is stored as:

```text
models/random_forest_model.pkl
```

The model can then be loaded by the prediction pipeline for EEG window-level classification.

The Random Forest implementation is considered a **baseline research model** and is not a clinically validated seizure detection model.

---

## 10. Model Evaluation

The experimental dataset was divided into training and testing subsets using an 80/20 split.

| Dataset | Samples |
|---|---:|
| Total Samples | 900 |
| Training Samples | 720 |
| Testing Samples | 180 |

The test set contained:

| Class | Samples |
|---|---:|
| Normal Samples | 178 |
| Seizure Samples | 2 |

The model was evaluated using:

- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- F1-Score
- Balanced Accuracy
- Confusion Matrix
- Classification Report

Because the dataset is severely imbalanced, accuracy alone is not considered sufficient for evaluating seizure detection performance.

---

## 11. Baseline Model Results

The baseline Random Forest model was initially evaluated using the default classification threshold:

```text
Threshold = 0.50
```

The model achieved high overall accuracy on the test set, but this result was strongly influenced by the large number of normal samples.

The baseline evaluation demonstrated the central challenge of this project:

```text
High overall accuracy
        +
Poor seizure detection
        =
Class imbalance problem
```

The initial baseline model failed to reliably detect the seizure class in the test split.

Therefore, the baseline result should not be interpreted as evidence of strong seizure detection performance.

The baseline confusion matrix is stored in:

```text
images/random_forest_baseline_confusion_matrix.png
```

---

## 12. Classification Threshold Analysis

The predicted seizure probabilities were further analyzed by changing the classification threshold.

The following thresholds were evaluated:

```text
0.50
0.20
0.10
0.05
0.01
```

The experimental results were:

| Threshold | Sensitivity | Specificity |
|---|---:|---:|
| 0.50 | 0.00% | 100.00% |
| 0.20 | 0.00% | 100.00% |
| 0.10 | 0.00% | 96.07% |
| 0.05 | 50.00% | 87.64% |
| 0.01 | 50.00% | 63.48% |

At a threshold of **0.05**, the experimental confusion matrix was:

| | Predicted Normal | Predicted Seizure |
|---|---:|---:|
| **Actual Normal** | 156 | 22 |
| **Actual Seizure** | 1 | 1 |

This corresponds to:

```text
Total Seizure Samples = 2
Detected Seizures     = 1
Sensitivity           = 50.00%

False Positives       = 22
Specificity           = 87.64%
```

This demonstrates the trade-off between increasing seizure sensitivity and increasing false-positive predictions.

The threshold analysis is exploratory because the test set contained only **two seizure samples**. Therefore, these results should not be considered statistically reliable threshold optimization.

The threshold analysis output is stored in:

```text
results/threshold_analysis.png
```

---

## 13. Event-Level Validation

An event-level validation was performed using:

```text
chb01_03.edf
```

The known seizure annotation for this recording indicates:

```text
Seizure Start = 2996 seconds
Seizure End   = 3036 seconds
```

The deployed Streamlit application processed the recording with:

```text
Channels            = 23
Sampling Frequency  = 256 Hz
Duration            = 3600 seconds
Window Duration     = 4 seconds
Total Windows       = 900
Operating Threshold = 0.40
```

The deployed application produced:

```text
Normal Windows       = 891
Seizure-like Windows = 9
```

The detected seizure-like windows were:

| Window | Time Interval | Seizure Probability |
|---|---|---:|
| 749 | 2996–3000 sec | 0.42 |
| 750 | 3000–3004 sec | 0.81 |
| 751 | 3004–3008 sec | 0.91 |
| 752 | 3008–3012 sec | 0.83 |
| 753 | 3012–3016 sec | 0.70 |
| 754 | 3016–3020 sec | 0.62 |
| 755 | 3020–3024 sec | 0.84 |
| 756 | 3024–3028 sec | 0.97 |
| 757 | 3028–3032 sec | 0.41 |

The first detected seizure-like window begins at **2996 seconds**, which matches the annotated seizure start time.

The detected sequence covers:

```text
2996–3032 seconds
```

while the annotated seizure extends to:

```text
3036 seconds
```

Because the prediction pipeline uses non-overlapping 4-second windows and an operating threshold of 0.40, the final interval from **3032–3036 seconds** was not classified as seizure-like.

The highest seizure probability observed in this recording was:

```text
0.97
```

for the:

```text
3024–3028 second window
```

### Interpretation

This result demonstrates that the deployed pipeline can identify a sequence of seizure-like EEG windows around the annotated seizure event in this particular recording.

However, this is **event-level validation on a single EEG recording**. It should not be interpreted as a statistically reliable estimate of overall model sensitivity, specificity, or clinical performance.

---

## 14. Streamlit Deployment

The prediction pipeline was deployed as an interactive **Streamlit** web application.

The application allows users to:

1. Upload an EEG EDF recording.
2. Inspect EEG recording information.
3. Preprocess the EEG signal.
4. Create 4-second EEG windows.
5. Extract features.
6. Run the trained Random Forest model.
7. Apply the selected operating threshold.
8. View normal and seizure-like window counts.
9. Inspect window-level predictions.
10. View the seizure detection timeline.
11. Download prediction results.

The deployed application successfully processed:

```text
chb01_03.edf
```

with:

```text
23 EEG channels
256 Hz sampling frequency
3600 seconds recording duration
900 complete EEG windows
```

The application is intended as a demonstration and research prototype.

> The deployed application is not a medical diagnostic tool and should not be used for clinical decision-making.

---

## 15. Key Findings

The main findings of the current experiments are:

- The project successfully implements an end-to-end EEG processing and machine learning workflow.
- The baseline Random Forest model achieved high overall accuracy, but accuracy was strongly affected by severe class imbalance.
- The baseline model did not reliably detect seizure samples at the default threshold.
- Threshold adjustment demonstrated the trade-off between seizure sensitivity and false-positive predictions.
- The experimental threshold results are unstable because only two seizure samples were present in the test set.
- The deployed application successfully processed a 1-hour, 23-channel EEG recording.
- The deployed pipeline created 900 complete 4-second windows.
- Using an operating threshold of 0.40, the application identified 9 seizure-like windows.
- On `chb01_03.edf`, the detected seizure-like activity began at 2996 seconds, matching the known seizure start time.
- The detected sequence extended from 2996 to 3032 seconds, while the annotated seizure ended at 3036 seconds.
- The highest seizure probability observed in the deployed test was 0.97.
- The results demonstrate the feasibility of an end-to-end research prototype while also highlighting the need for larger datasets and stronger validation.

---

## 16. Limitations

### 16.1 Severe Class Imbalance

The experimental dataset contains substantially more normal EEG windows than seizure windows.

```text
Normal Windows  = 890
Seizure Windows = 10
```

This imbalance makes it difficult for the model to learn robust seizure-specific patterns.

### 16.2 Limited Seizure Samples

Only a small number of seizure windows are available in the current experiment.

The test set contains only two seizure samples, making sensitivity estimates highly unstable.

### 16.3 Limited Dataset Scope

The current experiment uses a limited subset of the CHB-MIT dataset.

The model has not yet been extensively evaluated across multiple patients and recordings.

### 16.4 Patient-Independent Generalization

The current experiment does not provide sufficient evidence that the model will generalize to completely unseen patients.

Patient-independent evaluation is required before making stronger claims about generalization.

### 16.5 Threshold Analysis Limitations

The threshold analysis is based on a very small test set containing only two seizure samples.

Therefore, the threshold results are exploratory.

### 16.6 Single-Recording Event Validation

The deployed event-level validation was demonstrated using one EEG recording.

A single successful detection event does not establish general model performance.

### 16.7 Limited Feature Representation

The current model uses a relatively small feature set consisting of statistical and frequency-band features.

More advanced time-frequency and nonlinear features may improve representation of complex EEG patterns.

### 16.8 No Clinical Validation

This project is an educational and research prototype.

The model has not been clinically validated and should not be used for medical diagnosis, treatment, or clinical decision-making.

---

## 17. Future Work

Future improvements may include:

- Using a larger number of EEG recordings.
- Including data from multiple patients.
- Performing patient-independent evaluation.
- Increasing the number of seizure samples.
- Applying appropriate class imbalance handling techniques.
- Exploring class weighting and resampling methods.
- Performing stratified cross-validation.
- Optimizing classification thresholds using a dedicated validation set.
- Evaluating Precision-Recall curves.
- Evaluating ROC-AUC.
- Testing Support Vector Machines (SVM).
- Testing XGBoost and other ensemble methods.
- Exploring 1D CNN-based EEG classification.
- Exploring LSTM and other deep learning approaches.
- Extracting additional time-domain features.
- Extracting additional frequency-domain features.
- Investigating time-frequency representations such as wavelets.
- Evaluating performance across multiple patients.
- Improving seizure detection sensitivity while controlling false-positive rates.
- Investigating subject-specific and patient-independent seizure detection models.

---

## 18. Technologies Used

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

## 19. Project Structure

The repository contains the machine learning pipeline, deployment code, notebooks, model artifacts, and result files.

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
│   └── 10_Prediction.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── final_evaluation.txt
│   └── threshold_analysis.png
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

---

## 20. Installation

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

## 21. Running the Project Locally

### Run the Streamlit Application

From the project root:

```bash
streamlit run app.py
```

The application will open in your browser.

### Run the Notebook Workflow

The notebooks demonstrate the individual stages of the project pipeline.

The general sequence is:

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
```

---

## 22. Reproducibility

The project uses fixed random seeds where applicable to improve reproducibility of machine learning experiments.

Important processed artifacts include:

```text
data/features.csv
data/features.npy
data/labels.npy
models/random_forest_model.pkl
```

Evaluation outputs include:

```text
results/confusion_matrix.png
results/feature_importance.png
results/final_evaluation.txt
results/threshold_analysis.png
```

Raw EEG recordings are not stored in the repository and must be obtained separately from the CHB-MIT Scalp EEG Database.

---

## 23. Research Status

**Status: Research and Development / Baseline Prototype**

The project demonstrates an end-to-end pipeline covering:

- EEG data loading
- Signal preprocessing
- Window segmentation
- Seizure labeling
- Feature extraction
- Dataset construction
- Machine learning model training
- Model evaluation
- Classification threshold analysis
- Individual EEG window prediction
- Event-level detection
- Streamlit deployment

The current experimental results highlight the challenges of seizure detection under severe class imbalance.

The baseline model demonstrates the importance of using appropriate evaluation metrics rather than relying only on overall accuracy.

The deployed application demonstrates the complete prediction workflow on an EDF recording, including window-level predictions and a seizure detection timeline.

Future development will focus on improving seizure detection sensitivity, increasing the amount of seizure data, performing patient-independent evaluation, and investigating more robust machine learning and deep learning approaches.

---

## 24. Disclaimer

This project is developed for **educational and research purposes only**.

The current model is **not a clinically validated medical device** and should not be used for medical diagnosis, treatment, or clinical decision-making.

The reported results are based on limited experimental data and a limited event-level validation example. They should not be interpreted as evidence of clinical effectiveness.

---

## 25. Author

**Shivam Prajapati**

Computer Science Engineering — Artificial Intelligence & Machine Learning

**GitHub:**  
https://github.com/ShivamKumarPrajapati-123
