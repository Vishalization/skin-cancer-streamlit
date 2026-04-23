import streamlit as st
from PIL import Image

from model_loader import load_models
from utils import preprocess, interpret_prediction

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("Skin Cancer Detection using Deep Learning")
st.write("Upload a dermoscopic image for prediction.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Loading models and predicting..."):
        models = load_models()

        st.subheader("Predictions")

        for model_name, model in models.items():
            input_img = preprocess(img, model_name)

            preds = model.predict(input_img, verbose=0)

            class_name, confidence, label = interpret_prediction(preds)

            st.markdown(f"### {model_name}")
            st.write(f"Predicted Class: **{class_name}**")
            st.write(f"Confidence: **{confidence:.2f}**")

            if label == "Suspicious":
                st.error("⚠️ Suspicious")
            else:
                st.success("✅ Not Suspicious")

    st.info("This tool is for educational purposes only and not a medical diagnosis.")