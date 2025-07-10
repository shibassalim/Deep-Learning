import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
scaler = joblib.load("Scaler.pkl")
model = joblib.load("perceptron_model.pkl")

# Apply custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            background-color: #E8ECD6;
            color: #1F1F1F;
            font-family: 'Poppins', sans-serif;
        }

        .title-box {
            background-color: #B75B48;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            animation: pop 0.8s ease;
            color: white;
            font-size: 36px;
            font-weight: 600;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }

        @keyframes pop {
            0% { transform: scale(0.8); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .stButton>button {
            background-color: #B75B48;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1em;
            border: none;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #9e4a3a;
        }

        .stNumberInput input {
            background-color: #f6f6f6;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# Custom Title with Animation
st.markdown('<div class="title-box">BREAST CANCER PREDICTION</div>', unsafe_allow_html=True)

# Sidebar
page = st.sidebar.selectbox("Pages", options=["App", "About"])

# Main App Page
if page == "App":
    st.subheader("Enter the Feature Values:")

    radiusmean = st.number_input("Radius Mean")
    texturemean = st.number_input("Texture Mean")
    perimetermean = st.number_input("Perimeter Mean")
    areamean = st.number_input("Area Mean")
    smoothnessmean = st.number_input("Smoothness Mean")
    compactnessmean = st.number_input("Compactness Mean")
    concativitymean = st.number_input("Concavity Mean")
    concavepointsmean = st.number_input("Concave Points Mean")
    symmetrymean = st.number_input("Symmetry Mean")
    fractaldimmean = st.number_input("Fractal Dimension Mean")

    # DataFrame for prediction
    data = pd.DataFrame({
        "radius_mean": [radiusmean],
        "texture_mean": [texturemean],
        "perimeter_mean": [perimetermean],
        "area_mean": [areamean],
        "smoothness_mean": [smoothnessmean],
        "compactness_mean": [compactnessmean],
        "concavity_mean": [concativitymean],
        "concave points_mean": [concavepointsmean],
        "symmetry_mean": [symmetrymean],
        "fractal_dimension_mean": [fractaldimmean]
    })

    # Scale the input
    data_scaled = scaler.transform(data)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(data_scaled)[0]
        if prediction == 'M':
            st.error("Prediction: Malignant (Cancerous)")
        else:
            st.success("Prediction: Benign (Non-Cancerous)")

elif page == "About":
    st.subheader("About this App")
    st.write("""
        This Streamlit application predicts whether a breast tumor is **malignant** or **benign** 
        using a trained Perceptron model based on 10 mean cell feature values from the 
        Breast Cancer Wisconsin dataset.
    """)
