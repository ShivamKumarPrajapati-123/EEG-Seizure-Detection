import numpy as np
import pandas as pd
from scipy.signal import welch


# Frequency bands used during model training
FREQUENCY_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 40)
}


def extract_statistical_features(window):
    """
    Extract statistical features from one EEG window.

    Expected input shape:
        (23, 1024)

    Returns:
        Mean
        Standard Deviation
        Variance
    """

    # Calculate features for each EEG channel
    channel_means = np.mean(window, axis=1)
    channel_stds = np.std(window, axis=1)
    channel_variances = np.var(window, axis=1)

    # Average features across all EEG channels
    mean_feature = np.mean(channel_means)
    std_feature = np.mean(channel_stds)
    variance_feature = np.mean(channel_variances)

    return (
        mean_feature,
        std_feature,
        variance_feature
    )


def extract_frequency_features(window, sfreq=256):
    """
    Extract frequency-domain features using Welch PSD.

    Expected input shape:
        (23, 1024)

    Returns:
        Mean power in:
        Delta, Theta, Alpha, Beta, Gamma bands
    """

    channel_features = []

    # Process each EEG channel
    for channel in window:

        # Calculate Power Spectral Density
        frequencies, psd = welch(
            channel,
            fs=sfreq,
            nperseg=512
        )

        band_features = {}

        # Calculate power in each frequency band
        for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():

            frequency_mask = (
                (frequencies >= low_freq) &
                (frequencies < high_freq)
            )

            band_power = np.trapezoid(
                psd[frequency_mask],
                frequencies[frequency_mask]
            )

            band_features[band_name] = band_power

        channel_features.append(band_features)

    # Convert channel features to DataFrame
    channel_features_df = pd.DataFrame(channel_features)

    # Average frequency features across all EEG channels
    window_frequency_features = channel_features_df.mean()

    return window_frequency_features


def extract_all_features(window, sfreq=256):
    """
    Extract all 8 features used by the trained Random Forest model.

    Feature order:

    1. Mean
    2. Std
    3. Variance
    4. Delta
    5. Theta
    6. Alpha
    7. Beta
    8. Gamma
    """

    # Statistical features
    mean_feature, std_feature, variance_feature = (
        extract_statistical_features(window)
    )

    # Frequency-domain features
    frequency_features = extract_frequency_features(
        window,
        sfreq
    )

    # Combine features in the exact training order
    return np.array([
        mean_feature,
        std_feature,
        variance_feature,
        frequency_features["Delta"],
        frequency_features["Theta"],
        frequency_features["Alpha"],
        frequency_features["Beta"],
        frequency_features["Gamma"]
    ])


def extract_features_from_windows(windows, sfreq=256):
    """
    Extract features from multiple EEG windows.

    Input:
        windows shape = (number_of_windows, 23, 1024)

    Output:
        feature matrix shape = (number_of_windows, 8)
    """

    all_features = []

    for i, window in enumerate(windows):

        features = extract_all_features(
            window,
            sfreq
        )

        all_features.append(features)

    return np.array(all_features)