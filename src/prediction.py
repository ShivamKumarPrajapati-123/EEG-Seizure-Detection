import os
import joblib
import numpy as np

from src.feature_extraction import extract_all_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "random_forest_final.pkl"
)

# Final threshold selected during model evaluation
FINAL_THRESHOLD = 0.40


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

def load_model():
    """
    Load the trained Random Forest model.
    """

    model = joblib.load(MODEL_PATH)

    return model


# --------------------------------------------------
# Predict one EEG window
# --------------------------------------------------

def predict_window(
    window,
    model=None,
    sfreq=256,
    threshold=FINAL_THRESHOLD
):
    """
    Predict whether one EEG window is
    Normal or Seizure-like.

    Parameters
    ----------
    window : numpy.ndarray
        EEG window with shape:
        (23, 1024)

    model : trained model, optional
        Random Forest model.

    sfreq : int
        EEG sampling frequency.

    threshold : float
        Classification threshold.

    Returns
    -------
    result : dict
        Prediction information.
    """

    # Load model if not provided
    if model is None:
        model = load_model()

    # Extract the same 8 features used during training
    features = extract_all_features(
        window,
        sfreq
    )

    # Convert to 2D array
    # Required shape:
    # (1, 8)

    features = np.array(
        features
    ).reshape(1, -1)

    # Get seizure probability
    seizure_probability = model.predict_proba(
        features
    )[0][1]

    # Apply final threshold
    if seizure_probability >= threshold:

        predicted_class = 1
        predicted_label = "Seizure"

    else:

        predicted_class = 0
        predicted_label = "Normal"

    return {
        "seizure_probability": float(
            seizure_probability
        ),
        "threshold": float(
            threshold
        ),
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "features": features
    }