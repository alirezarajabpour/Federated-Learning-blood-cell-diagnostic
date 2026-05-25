import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
# import time
import pandas as pd
# import numpy as np

from model import get_model
from dataset import CLASS_NAMES, prepare_client_datasets, load_data

MODEL_PATH = "final_global_model.pth"
NUM_CLASSES = 8

st.set_page_config(
    page_title="Federated Diagnostic Center",
    page_icon="🩸",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Loads the trained global model from the specified path."""
    if not os.path.exists(MODEL_PATH):
        return None
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model


st.title("🩸 Federated Diagnostic Center")
st.markdown(
    "An AI-powered diagnostic tool built using **Federated Learning**. This model was trained collaboratively across multiple simulated hospitals without sharing any private patient data, leveraging an advanced **Personalization Strategy (MAP)** to overcome challenges of incomplete datasets."
)
st.markdown("---")


st.header("Our Collaborating Federation")
st.write("Each simulated hospital in our network contributes its unique, private data. A key challenge is that no single hospital has examples of all 8 blood cell types, which makes standard training difficult. This is the problem our MAP strategy solves.")

train_dataset, _, num_classes = load_data()
_, client_class_map = prepare_client_datasets(train_dataset, num_classes)

hospital_names = [
    "St. Jude's Research Hospital",
    "General Hospital of Hematology",
    "The Institute of Cellular Pathology",
    "Mercy Health Center",
    "City Central Medical",
    "Oak Valley Clinic",
    "Pine Ridge Diagnostics",
    "Maple Leaf Institute",
    "Cedar Sinai Medical",
    "University Medical Group"
]

cols = st.columns(len(client_class_map))
for i, col in enumerate(cols):
    with col:
        st.subheader(f"Hospital {i+1}")
        st.markdown(f"*{hospital_names[i]}*")
        classes = client_class_map.get(i, [])
        st.metric(label="Known Cell Types", value=f"{len(classes)} / {NUM_CLASSES}")
        # Display the list of classes this hospital has
        class_list_str = ", ".join([CLASS_NAMES[c] for c in classes])
        st.info(f"Specializes in: {class_list_str}")

st.markdown("---")


st.header("🔬 Live Diagnostic Tool")
model = load_model()

if model is None:
    st.warning("Model file not found. Please wait for the federated learning process to complete, or run it if you haven't yet. The page will check for the model automatically.")
    if st.button("Check for Model Again"):
        st.rerun()
else:
    st.success("✅ **Trained Federated Model is Online.** Please upload a cell image.")

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

    with col2:
        if uploaded_file:
            with st.spinner("Analyzing image..."):
                transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ])
                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    confidence, predicted_idx = torch.max(probabilities, 0)

                    predicted_class = CLASS_NAMES.get(predicted_idx.item(), "Unknown")
                    confidence_percent = confidence.item() * 100

            st.metric(label="**Diagnosis Result**", value=predicted_class)
            st.metric(label="**Confidence**", value=f"{confidence_percent:.2f}%")

            st.subheader("Full Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Cell Type": [CLASS_NAMES.get(i, "Unknown") for i in range(NUM_CLASSES)],
                "Probability": probabilities.cpu().numpy()
            })
            st.bar_chart(prob_df, x="Cell Type", y="Probability")
