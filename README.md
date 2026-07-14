
# EEG-Based Epileptic Seizure Detection Using Signal Processing and Machine Learning

## Project Overview

Epileptic seizure are neurological events caused by abnormal electrical activity in the brain . Early and accurate seizure detection is important for diagnosis, patient monitoring, and timely medical intervention.

This project presents an end-to-end machine learning pipeline for epilepti seizure detection using electroencephalogram (EEG) signals from the CHB-MIT Scalp EEG Database. The raw EEG recordings are preprocessed, divided into fixed-length windows, labeled according to seizure intervals, and transformed into statistical and frequency-domin features. These features are then used to train and evaluate a Random Forest classifier for seizure detection.

The project demonstrates the complete workflow of biomedical signal processing, feature engineering, machine learning model development, and performance evaluation using python.

----

## Key Features

-Read raw EEG (.edf) recordings using MNE-Python

-Apply EEG signal preprocessing

-Generate fixed-length EEG windows

-Label seizure and non-seizure windows

-Extract statistical and spectral features

-Train a Random Forest classifier

-Evaluate model performance using standard machine learning metrics

-Save the trained model for future inference

----

## Problem Statement

Epilepsy is one of the most common neurological disorders worldwide. Neurologists diagnose epileptic seizures by analyzing Electroencephalogram (EEG) recordings, which is often a time-consuming and expertise-dependent process.

The objective of this project is to develop an automated machine learning system capable of distinguishing seizure and non-seizure EEG segments using statistical and frequency-domin features extracted from EEG recordings.

The project aims to demonstrate how signal processing and machine learning techniques can support computer-aided diagnosis in healthcare.

## Objectives

The primary objectives of this project are:

-Read raw EEG recordings from the CHB-MIT Scalp EEG Database.

-Apply preprocessing techniques to reduce signal noise.

-Divide EEG recordings into fixed-length windows.

-Label EEG windows as seizure or non-seizure.

Extract statistical and spectral features.

Build a machine learning dataset.

Train a Random Forest classifier.

Evaluate the model using stadard classification metrics.

Save the trained model for future predictions.


## Dataset

This project uses the **CHB-MIT Scalp EEG Database**, a publicly available dataset provided by the Massachusetts Institute of Technology (MIT) and Boston Children's Hospital.

### Dataset Information

- Dataset: CHB-MIT Scalp EEG Database
- Patient Used: chb01
- Recording Format: EDF (European Data Format)
- Sampling Frequency: 256 Hz
- Number of EEG Channels: 23
- Window Length: 4 seconds

### Files Used

- chb01_01.edf
- chb01_03.edf
- chb01_04.edf
- chb01_09.edf
- chb01_15.edf
- chb01_18.edf
- chb01_21.edf
- chb01_26.edf
- chb01_30.edf
- chb01_38.edf
- chb01_39.edf
- chb01_40.edf
- chb01_41.edf
- chb01_42.edf
- chb01_46.edf

### Dataset Summary

- Total Windows: 900
- Normal Windows: 889
- Seizure Windows: 11

## Technologies Used

### Programming Language

- Python 3

### Libraries

- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- MNE-Python
- Joblib

### Machine Learning Algorithm

- Random Forest Classifier

### Development Environment

- Jupyter Notebook
- Visual Studio Code
- 
- 