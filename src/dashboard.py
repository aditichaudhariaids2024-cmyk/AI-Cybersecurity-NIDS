import streamlit as st
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"
DATA_FILE = "data/training/nids_training_dataset.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Cybersecurity Threat Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ AI-Powered Cybersecurity Threat Detection")

st.write(
    "Network Intrusion Detection System using "
    "CIC-IDS2017 and Random Forest"
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_sample_data():

    df = pd.read_csv(
        DATA_FILE,
        nrows=10000
    )

    return df


try:

    df = load_sample_data()

except Exception as e:

    st.error(f"Unable to load dataset: {e}")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("System Information")

st.sidebar.write("**Dataset:** CIC-IDS2017")
st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**Features:** 80")
st.sidebar.write("**Training Samples:** 500,000")

st.sidebar.divider()

st.sidebar.write("**API:** FastAPI")
st.sidebar.write("**Status:** Local")


# ============================================================
# DATASET STATISTICS
# ============================================================

total_samples = len(df)

benign_count = len(
    df[df["Target"] == 0]
)

attack_count = len(
    df[df["Target"] == 1]
)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Samples Loaded",
        f"{total_samples:,}"
    )

with col2:

    st.metric(
        "BENIGN",
        f"{benign_count:,}"
    )

with col3:

    st.metric(
        "ATTACK",
        f"{attack_count:,}"
    )


st.divider()


# ============================================================
# TRAFFIC ANALYZER
# ============================================================

st.header("🔍 Network Traffic Analyzer")

st.write(
    "Select a network-flow sample from the CIC-IDS2017 dataset "
    "and send it to the trained Random Forest model."
)


# Create readable sample numbers

sample_index = st.number_input(
    "Select Sample",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)


selected_row = df.iloc[sample_index]

actual_target = int(
    selected_row["Target"]
)


# ============================================================
# DISPLAY ACTUAL CLASS
# ============================================================

if actual_target == 0:

    st.info(
        "Actual dataset class: BENIGN"
    )

else:

    st.warning(
        "Actual dataset class: ATTACK"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔎 Analyze Traffic",
    type="primary"
):

    features = selected_row.drop(
        labels=["Target"]
    ).to_dict()

    try:

        response = requests.post(
            API_URL,
            json={
                "features": features
            },
            timeout=30
        )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]

            attack_probability = (
                result["attack_probability"] * 100
            )

            benign_probability = (
                result["benign_probability"] * 100
            )


            st.divider()

            st.header("🚨 Detection Result")


            # ------------------------------------------------
            # ATTACK
            # ------------------------------------------------

            if prediction == "ATTACK":

                st.error(
                    "⚠️ POTENTIAL NETWORK INTRUSION DETECTED"
                )

            # ------------------------------------------------
            # BENIGN
            # ------------------------------------------------

            else:

                st.success(
                    "✅ TRAFFIC CLASSIFIED AS BENIGN"
                )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Attack Probability",
                    f"{attack_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Benign Probability",
                    f"{benign_probability:.2f}%"
                )


            # ------------------------------------------------
            # MESSAGE
            # ------------------------------------------------

            st.info(
                result["message"]
            )


            # ------------------------------------------------
            # COMPARE WITH ACTUAL VALUE
            # ------------------------------------------------

            predicted_target = result["target"]

            if predicted_target == actual_target:

                st.success(
                    "Prediction matches the dataset label."
                )

            else:

                st.warning(
                    "Prediction differs from the dataset label."
                )


        else:

            st.error(
                f"API Error: {response.text}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Make sure the FastAPI server is running."
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )


# ============================================================
# DATA PREVIEW
# ============================================================

st.divider()

st.header("📊 Network Traffic Sample")

display_columns = [
    "Source Port",
    "Destination Port",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Average Packet Size",
    "Target"
]

available_columns = [
    column
    for column in display_columns
    if column in df.columns
]

st.dataframe(
    df[available_columns].head(20),
    use_container_width=True
)