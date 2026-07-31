import os
import streamlit as st
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from src.preprocessing import create_windows
from src.feature_extraction import extract_features_from_windows
from src.prediction import load_model, FINAL_THRESHOLD


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EEG Seizure Detection Research System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    .research-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #d9d9d9;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dddddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 EEG-Based Seizure Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An end-to-end EEG signal processing and machine learning research prototype'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    This system processes EEG recordings in EDF format and performs
    window-level seizure-like activity detection using signal processing,
    feature extraction, and a trained Random Forest classifier.
    """
)


st.warning(
    "⚠️ Research Prototype: This system is intended for educational and "
    "research purposes only. It is not a clinically validated medical "
    "diagnostic system and must not be used for medical decision-making."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧪 Research Prototype")

    st.markdown(
        """
        **Pipeline**

        1. EDF EEG Upload
        2. EEG Signal Loading
        3. Signal Preprocessing
        4. 4-Second Windowing
        5. Feature Extraction
        6. Random Forest Prediction
        7. Threshold-Based Classification
        8. Seizure-Like Activity Timeline
        """
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Model:** Random Forest")
    st.write("**Window Duration:** 4 seconds")
    st.write("**Expected Sampling Frequency:** 256 Hz")
    st.write(f"**Operating Threshold:** {FINAL_THRESHOLD}")

    st.divider()

    st.caption(
        "Developed by Shivam Prajapati"
    )

    st.caption(
        "Computer Science Engineering — Artificial Intelligence & Machine Learning"
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    return load_model()


try:

    model = get_model()

except Exception as e:

    st.error(
        "Unable to load the trained machine learning model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("1. Upload EEG Recording")

uploaded_file = st.file_uploader(
    "Upload an EEG recording in EDF format",
    type=["edf"],
    help="Upload a CHB-MIT style EEG EDF recording for analysis."
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.success(
        f"EEG recording uploaded successfully: {uploaded_file.name}"
    )

    temp_file = "temp_uploaded.edf"


    # --------------------------------------------------------
    # SAVE UPLOADED FILE
    # --------------------------------------------------------

    with open(temp_file, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze_button = st.button(
        "🔍 Analyze EEG Recording",
        type="primary",
        use_container_width=True
    )


    if analyze_button:

        try:

            # ==================================================
            # LOAD EEG
            # ==================================================

            with st.spinner(
                "Loading EEG recording..."
            ):

                raw = mne.io.read_raw_edf(
                    temp_file,
                    preload=True,
                    verbose=False
                )


            # ==================================================
            # RECORDING INFORMATION
            # ==================================================

            st.header("2. EEG Recording Information")


            actual_sfreq = float(
                raw.info["sfreq"]
            )

            duration = float(
                raw.times[-1]
            )

            channel_count = len(
                raw.ch_names
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "EEG Channels",
                    channel_count
                )


            with col2:

                st.metric(
                    "Sampling Frequency",
                    f"{actual_sfreq:.1f} Hz"
                )


            with col3:

                st.metric(
                    "Recording Duration",
                    f"{duration / 60:.2f} min"
                )


            with col4:

                st.metric(
                    "Duration",
                    f"{duration:.0f} sec"
                )


            with st.expander(
                "View EEG Channel Names"
            ):

                st.write(
                    raw.ch_names
                )


            # ==================================================
            # GET EEG DATA
            # ==================================================

            with st.spinner(
                "Preparing EEG signal..."
            ):

                eeg_data = raw.get_data()


            # ==================================================
            # WINDOWING
            # ==================================================

            st.header("3. EEG Windowing")


            window_duration = 4


            with st.spinner(
                "Dividing EEG recording into 4-second windows..."
            ):

                windows = create_windows(
                    eeg_data,
                    sfreq=actual_sfreq,
                    window_duration=window_duration
                )


            total_windows = len(
                windows
            )


            st.success(
                f"Successfully created {total_windows} complete "
                f"{window_duration}-second EEG windows."
            )


            # ==================================================
            # FEATURE EXTRACTION
            # ==================================================

            st.header("4. EEG Feature Extraction")


            with st.spinner(
                "Extracting statistical and frequency-domain features..."
            ):

                features = extract_features_from_windows(
                    windows,
                    sfreq=actual_sfreq
                )


            st.success(
                f"Feature extraction completed successfully. "
                f"Feature matrix shape: {features.shape}"
            )


            # ==================================================
            # MODEL PREDICTION
            # ==================================================

            st.header("5. Machine Learning Prediction")


            with st.spinner(
                "Running Random Forest seizure detection..."
            ):

                probabilities = (
                    model.predict_proba(
                        features
                    )[:, 1]
                )


            predictions = (
                probabilities >= FINAL_THRESHOLD
            ).astype(int)


            # ==================================================
            # RESULT COUNTS
            # ==================================================

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


            seizure_percentage = (

                seizure_windows
                / total_windows
                * 100

                if total_windows > 0

                else 0

            )


            # ==================================================
            # SUMMARY METRICS
            # ==================================================

            st.subheader(
                "Detection Summary"
            )


            col1, col2, col3, col4 = st.columns(4)


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
                    "Seizure-Like Windows",
                    seizure_windows
                )


            with col4:

                st.metric(
                    "Detected Window %",
                    f"{seizure_percentage:.2f}%"
                )


            st.info(
                f"Operating classification threshold: "
                f"**{FINAL_THRESHOLD:.2f}**"
            )


            # ==================================================
            # OVERALL DETECTION RESULT
            # ==================================================

            if seizure_windows > 0:

                st.error(
                    f"⚠️ Seizure-like activity detected in "
                    f"**{seizure_windows}** EEG window(s)."
                )

            else:

                st.success(
                    "✅ No seizure-like EEG windows were detected "
                    "using the current operating threshold."
                )


            # ==================================================
            # CREATE RESULTS DATAFRAME
            # ==================================================

            results = []


            for i in range(
                total_windows
            ):

                start_time = (
                    i
                    * window_duration
                )


                end_time = (
                    start_time
                    + window_duration
                )


                probability = float(
                    probabilities[i]
                )


                prediction = (

                    "Seizure-like"

                    if predictions[i] == 1

                    else "Normal"

                )


                results.append(

                    {

                        "Window":

                            i,

                        "Start Time (sec)":

                            start_time,

                        "End Time (sec)":

                            end_time,

                        "Seizure Probability":

                            round(
                                probability,
                                4
                            ),

                        "Prediction":

                            prediction

                    }

                )


            results_df = pd.DataFrame(
                results
            )


            # ==================================================
            # PROBABILITY TIMELINE
            # ==================================================

            st.header(
                "6. Seizure Probability Timeline"
            )


            st.markdown(
                """
                The graph below shows the predicted seizure probability
                for each consecutive 4-second EEG window.

                The horizontal threshold represents the operating
                classification threshold used by the model.
                """
            )


            fig, ax = plt.subplots(
                figsize=(14, 5)
            )


            time_axis = (

                results_df[
                    "Start Time (sec)"
                ]

            )


            probability_axis = (

                results_df[
                    "Seizure Probability"
                ]

            )


            ax.plot(

                time_axis,

                probability_axis,

                linewidth=1.2,

                label="Seizure Probability"

            )


            ax.axhline(

                y=FINAL_THRESHOLD,

                linestyle="--",

                linewidth=1.5,

                label=f"Threshold = {FINAL_THRESHOLD:.2f}"

            )


            ax.set_xlabel(
                "Time (seconds)"
            )


            ax.set_ylabel(
                "Seizure Probability"
            )


            ax.set_title(
                "EEG Seizure Probability Across Recording"
            )


            ax.set_ylim(
                0,
                1
            )


            ax.grid(
                alpha=0.3
            )


            ax.legend()


            st.pyplot(
                fig
            )


            plt.close(
                fig
            )


            # ==================================================
            # WINDOW-LEVEL PREDICTIONS
            # ==================================================

            st.header(
                "7. Window-Level Predictions"
            )


            show_seizure_only = st.checkbox(

                "Show only seizure-like windows",

                value=False

            )


            if show_seizure_only:

                filtered_results = (

                    results_df[

                        results_df[
                            "Prediction"
                        ]

                        == "Seizure-like"

                    ]

                )

            else:

                filtered_results = (

                    results_df

                )


            st.dataframe(

                filtered_results,

                use_container_width=True,

                hide_index=True

            )


            # ==================================================
            # DETECTED SEIZURE-LIKE WINDOWS
            # ==================================================

            st.header(
                "8. Detected Seizure-Like Activity"
            )


            seizure_results = (

                results_df[

                    results_df[
                        "Prediction"
                    ]

                    == "Seizure-like"

                ]

            )


            if len(
                seizure_results
            ) > 0:

                st.write(

                    f"The model identified "
                    f"**{len(seizure_results)}** "
                    f"seizure-like windows."

                )


                for _, row in (

                    seizure_results.iterrows()

                ):

                    st.warning(

                        f"Window {int(row['Window'])} | "

                        f"{row['Start Time (sec)']:.0f}–"

                        f"{row['End Time (sec)']:.0f} sec | "

                        f"Probability: "

                        f"{row['Seizure Probability']:.4f}"

                    )


            else:

                st.success(

                    "No seizure-like windows were detected."

                )


            # ==================================================
            # DOWNLOAD RESULTS
            # ==================================================

            st.header(
                "9. Download Analysis Results"
            )


            csv_data = (

                results_df.to_csv(
                    index=False
                )

            )


            st.download_button(

                label=
                    "📥 Download Window-Level Predictions (CSV)",

                data=
                    csv_data,

                file_name=
                    "eeg_seizure_predictions.csv",

                mime=
                    "text/csv",

                use_container_width=True

            )


            # ==================================================
            # RESEARCH INTERPRETATION
            # ==================================================

            st.header(
                "10. Research Interpretation"
            )


            st.markdown(

                f"""
                **Recording analyzed:** `{uploaded_file.name}`

                **EEG channels:** {channel_count}

                **Sampling frequency:** {actual_sfreq:.1f} Hz

                **Recording duration:** {duration:.2f} seconds

                **Window duration:** {window_duration} seconds

                **Total windows analyzed:** {total_windows}

                **Seizure-like windows:** {seizure_windows}

                **Operating threshold:** {FINAL_THRESHOLD:.2f}

                This output represents window-level model predictions
                from the research prototype. A seizure-like prediction
                does not constitute a medical diagnosis.
                """

            )


            # ==================================================
            # CLEANUP
            # ==================================================

            if os.path.exists(
                temp_file
            ):

                os.remove(
                    temp_file
                )


        except Exception as e:

            st.error(

                "An error occurred while processing "
                "the EEG recording."

            )


            st.exception(
                e
            )

else:

    st.info(

        "👆 Upload an EDF EEG recording above to begin analysis."

    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(

    "EEG-Based Epileptic Seizure Detection | "
    "Research & Learning Prototype | "
    "Not for Clinical Diagnosis"

)