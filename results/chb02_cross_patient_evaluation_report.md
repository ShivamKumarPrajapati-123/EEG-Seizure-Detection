
# CHB01 → CHB02 Cross-Patient EEG Seizure Detection

## Evaluation Setup

- Training patient: **CHB01**
- Unseen test patient: **CHB02**
- Window duration: **4 seconds**
- Decision threshold: **0.05**
- Threshold selection: CHB01 internal validation
- CHB02 used only for final evaluation
- Model type: Random Forest

## Window-Level Results

| Metric | Result |
|---|---:|
| Test windows | 6539 |
| Actual seizure windows | 45 |
| Predicted seizure windows | 385 |
| True positives | 30 |
| False positives | 355 |
| True negatives | 6139 |
| False negatives | 15 |
| Accuracy | 0.9434 |
| Sensitivity | 0.6667 |
| Specificity | 0.9453 |
| Precision | 0.0779 |
| F1-score | 0.1395 |
| ROC-AUC | 0.9308 |

## Event-Level Results

| Metric | Result |
|---|---:|
| Actual annotated seizures | 3 |
| Detected annotated seizures | 3 |
| Missed annotated seizures | 0 |
| Predicted seizure events | 201 |
| Non-overlapping predicted events | 195 |
| False-positive events/hour | 37.03 |

## Seizure Coverage

| Metric | Result |
|---|---:|
| Total annotated seizure duration | 172 s |
| Total covered seizure duration | 114 s |
| Total missed seizure duration | 58 s |
| Simple average coverage | 76.24% |
| Time-weighted coverage | 66.28% |
| Average detection delay | 3.33 s |
| Number of missed coverage gaps | 5 |

## Interpretation

The CHB01-trained Random Forest achieved a ROC-AUC of **0.9308**
when evaluated on the completely unseen CHB02 patient.

Using the decision threshold of **0.05**, which was selected using
CHB01 internal validation, the detector produced:

- **66.67% sensitivity**
- **94.53% specificity**
- **7.79% precision**
- **13.95% F1-score**

At the event level, all **3 annotated CHB02 seizure events were
overlapped by at least one predicted seizure window**.

The predicted windows covered **66.28% of the total annotated
seizure duration**, corresponding to 114 of 172 annotated seizure
seconds.

However, the detector generated **195 predicted events that did
not overlap an annotated seizure**, corresponding to approximately
**37.03 false-positive events per hour**.

These results indicate that the model has useful cross-patient
ranking and seizure-event detection capability, but the current
feature set and decision threshold also produce a substantial
false-positive burden.

## Important Methodological Note

The CHB02 patient was treated as an unseen evaluation patient.

The threshold was **not optimized using CHB02**.

The threshold of **0.05** was selected from CHB01 internal
validation before applying the model to CHB02.

The threshold comparison performed on CHB02 is therefore considered
diagnostic analysis rather than model-selection evidence.

## Limitations

1. Only one unseen patient (CHB02) was evaluated in this
   cross-patient experiment.
2. CHB02 contains only three annotated seizure events in the
   evaluated recordings.
3. The dataset is highly imbalanced between normal and seizure
   windows.
4. The current detector produces a substantial false-positive
   burden.
5. The current model uses a relatively small handcrafted feature
   set.
6. The experiment is research/experimental evaluation and does not
   constitute clinical validation.

## Future Work

Potential improvements include:

- evaluation across additional CHB-MIT patients
- patient-independent cross-validation
- improved temporal post-processing
- richer time-frequency EEG features
- additional class-imbalance strategies
- comparison with alternative machine-learning models
- systematic evaluation of false-alarm reduction
- more extensive patient-independent testing

