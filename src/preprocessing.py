import numpy as np


def create_windows(
    eeg_data,
    sfreq=256,
    window_duration=4
):
    """
    Divide continuous EEG data into fixed-length windows.

    Parameters
    ----------
    eeg_data : numpy.ndarray
        EEG data with shape:
        (channels, samples)

    sfreq : int
        Sampling frequency in Hz.

    window_duration : int
        Window duration in seconds.

    Returns
    -------
    windows : numpy.ndarray
        Windowed EEG data with shape:
        (number_of_windows, channels, samples_per_window)
    """

    # Number of samples in each window
    samples_per_window = int(
        sfreq * window_duration
    )

    # Total number of complete windows
    n_windows = (
        eeg_data.shape[1] // samples_per_window
    )

    # Ignore incomplete data at the end
    usable_samples = (
        n_windows * samples_per_window
    )

    eeg_data = eeg_data[
        :, :usable_samples
    ]

    # Create windows
    windows = eeg_data.reshape(
        eeg_data.shape[0],
        n_windows,
        samples_per_window
    )

    # Rearrange dimensions:
    # (channels, windows, samples)
    # →
    # (windows, channels, samples)

    windows = np.transpose(
        windows,
        (1, 0, 2)
    )

    return windows