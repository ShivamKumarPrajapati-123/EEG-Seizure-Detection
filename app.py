import streamlit as st
import numpy as np
import pandas as pd
import mne

from src.preprocessing import create_windows
from src.feature_extraction import extract_features_from_windows
from src.prediction import load_model, FINAL_THRESHOLD


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="EEG Seizure Detection",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🧠 EEG Seizure Detection System")

st.markdown(
    """
    This application analyzes EEG recordings and classifies
    4-second EEG windows as **Normal** or **Seizure-like**.

    **Important:** This is a research prototype and is not
    a clinically validated medical diagnostic system.
    """
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


model = get_model()


# --------------------------------------------------
# File Upload
# --------------------------------------------------

st.header("1. Upload EEG Recording")

uploaded_file = st.file_uploader(
    "Upload an EEG EDF file",
    type=["edf"]
)


# --------------------------------------------------
# Analyze EEG
# --------------------------------------------------

if uploaded_file is not None:

    st.success(
        f"File uploaded: {uploaded_file.name}"
    )

    temp_file = "temp_uploaded.edf"

    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # --------------------------------------------------
    # Analyze Button
    # --------------------------------------------------

    if st.button(
        "🔍 Analyze EEG",
        type="primary"
    ):

        with st.spinner(
            "Loading and analyzing EEG signal..."
        ):

            try:

                # ------------------------------------------
                # Load EEG
                # ------------------------------------------

                raw = mne.io.read_raw_edf(
                    temp_file,
                    preload=True,
                    verbose=False
                )


                # ------------------------------------------
                # EEG Recording Information
                # ------------------------------------------

                st.header(
                    "2. EEG Recording Information"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Channels",
                        len(raw.ch_names)
                    )

                with col2:
                    st.metric(
                        "Sampling Frequency",
                        f"{raw.info['sfreq']} Hz"
                    )

                with col3:
                    st.metric(
                        "Duration",
                        f"{raw.times[-1]:.2f} sec"
                    )


                # ------------------------------------------
                # Get EEG Data
                # ------------------------------------------

                eeg_data = raw.get_data()


                # ------------------------------------------
                # Create 4-second Windows
                # ------------------------------------------

                windows = create_windows(
                    eeg_data,
                    sfreq=256,
                    window_duration=4
                )

                st.info(
                    f"Created {len(windows)} complete "
                    f"4-second EEG windows."
                )


                # ------------------------------------------
                # Feature Extraction
                # ------------------------------------------

                with st.spinner(
                    "Extracting EEG features..."
                ):

                    features = (
                        extract_features_from_windows(
                            windows,
                            sfreq=256
                        )
                    )

                st.success(
                    "Feature extraction completed."
                )


                # ------------------------------------------
                # Model Prediction
                # ------------------------------------------

                with st.spinner(
                    "Running seizure detection model..."
                ):

                    probabilities = (
                        model.predict_proba(
                            features
                        )[:, 1]
                    )


                # Apply final threshold
                predictions = (
                    probabilities >= FINAL_THRESHOLD
                ).astype(int)


                # ------------------------------------------
                # Count Predictions
                # ------------------------------------------

                total_windows = len(
                    predictions
                )

                seizure_windows = int(
                    np.sum(
                        predictions == 1
                    )
                )

                normal_windows = int(
                    np.sum(
                        predictions == 0
                    )
                )


                # ------------------------------------------
                # Prediction Results
                # ------------------------------------------

                st.header(
                    "3. Prediction Results"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Windows",
                        total_windows
                    )

                with col2:
                    st.metric(
                        "Normal Windows",
                        normal_windows
                    )

                with col3:
                    st.metric(
                        "Seizure-like Windows",
                        seizure_windows
                    )


                st.write(
                    f"Operating Threshold: "
                    f"**{FINAL_THRESHOLD}**"
                )


                # ------------------------------------------
                # Overall Result
                # ------------------------------------------

                if seizure_windows > 0:

                    st.error(
                        f"⚠️ Seizure-like activity detected "
                        f"in {seizure_windows} EEG window(s)."
                    )

                else:

                    st.success(
                        "✅ No seizure-like activity "
                        "was detected in the analyzed windows."
                    )


                # ------------------------------------------
                # Create Prediction DataFrame
                # ------------------------------------------

                results = []

                for i in range(
                    total_windows
                ):

                    start_time = i * 4

                    end_time = start_time + 4

                    label = (
                        "Seizure"
                        if predictions[i] == 1
                        else "Normal"
                    )

                    results.append({

                        "Window": i,

                        "Start Time (sec)": start_time,

                        "End Time (sec)": end_time,

                        "Seizure Probability": round(
                            float(
                                probabilities[i]
                            ),
                            4
                        ),

                        "Prediction": label

                    })


                results_df = pd.DataFrame(
                    results
                )


                # ------------------------------------------
                # Window-Level Predictions
                # ------------------------------------------

                st.header(
                    "4. Window-Level Predictions"
                )


                # Filter option
                show_seizure_only = st.checkbox(
                    "Show only seizure-like windows"
                )


                if show_seizure_only:

                    filtered_results = results_df[
                        results_df["Prediction"] == "Seizure"
                    ]

                else:

                    filtered_results = results_df


                st.dataframe(
                    filtered_results,
                    width="stretch"
                )


                # ------------------------------------------
                # Seizure Detection Timeline
                # ------------------------------------------

                st.header(
                    "5. Seizure Detection Timeline"
                )


                if seizure_windows > 0:

                    seizure_results = results_df[
                        results_df["Prediction"] == "Seizure"
                    ]


                    st.write(
                        "Detected seizure-like activity "
                        "at the following time intervals:"
                    )


                    for _, row in seizure_results.iterrows():

                        st.warning(
                            f"Window {int(row['Window'])}: "
                            f"{row['Start Time (sec)']:.0f}–"
                            f"{row['End Time (sec)']:.0f} seconds | "
                            f"Seizure Probability: "
                            f"{row['Seizure Probability']:.4f}"
                        )

                else:

                    st.success(
                        "No seizure-like windows were detected."
                    )


                # ------------------------------------------
                # Download Results
                # ------------------------------------------

                st.header(
                    "6. Download Results"
                )


                csv_data = results_df.to_csv(
                    index=False
                )


                st.download_button(

                    label=(
                        "📥 Download Prediction Results"
                    ),

                    data=csv_data,

                    file_name=(
                        "eeg_seizure_predictions.csv"
                    ),

                    mime="text/csv"

                )


            except Exception as e:

                st.error(
                    "An error occurred while "
                    "processing the EEG file."
                )

                st.exception(e)