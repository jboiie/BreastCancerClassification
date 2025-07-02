import torch
import torch.nn as nn
import streamlit as st
import numpy as np
import pandas as pd

# Define feature names (model expects them in order)
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# UI display names
display_names = [f.replace("_", " ").capitalize() for f in feature_names]

# Define model class
class CancerNet(nn.Module):
    def __init__(self):
        super(CancerNet, self).__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x

# Load the trained model
model = CancerNet()
model.load_state_dict(torch.load("cancer_model.pth", map_location=torch.device('cpu')))
model.eval()

# Streamlit config
st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")
st.markdown("""
    <style>
    body { background-color: #0e0e0e; color: #d0e6c2; }
    .stApp { background-color: #0e0e0e; }
    h1, h2, h3, h4, h5, h6 { color: #a4c79f; }
    .stButton>button { background-color: #4f6f52; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Breast Cancer Diagnosis Prediction")
st.write("Predict whether a tumor is benign or malignant using patient data.")

# Radio switch for input mode
mode = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

if mode == "Manual Input":
    st.subheader("🔢 Enter Feature Values")
    input_data = []
    cols = st.columns(2)
    for i, (feature, display_name) in enumerate(zip(feature_names, display_names)):
        value = cols[i % 2].number_input(display_name, value=0.0)
        input_data.append(value)

    if st.button("Predict from Manual Input"):
        with torch.no_grad():
            input_tensor = torch.tensor([input_data], dtype=torch.float32)
            output = model(input_tensor).item()
            prediction = "Malignant" if output >= 0.5 else "Benign"
            st.subheader("Prediction:")
            st.write(f"🔍 The tumor is likely **{prediction}** with a probability of `{output:.4f}`")

else:
    st.subheader("📄 Upload a CSV File")
    uploaded_file = st.file_uploader("Upload a CSV with exactly 30 features", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if all(f in df.columns for f in feature_names):
                st.success("CSV loaded successfully!")

                if st.button("Predict from CSV"):
                    with torch.no_grad():
                        inputs = torch.tensor(df[feature_names].values, dtype=torch.float32)
                        outputs = model(inputs).squeeze().numpy()
                        predictions = np.where(outputs >= 0.5, "Malignant", "Benign")

                        results = df.copy()
                        results["Prediction"] = predictions
                        results["Probability"] = outputs.round(4)

                        st.subheader("Prediction Results:")
                        st.dataframe(results)

                        csv_download = results.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Results as CSV", data=csv_download, file_name="cancer_predictions.csv", mime="text/csv")

            else:
                st.error("The uploaded CSV does not contain the required 30 features.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
