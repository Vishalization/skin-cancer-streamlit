import streamlit as st
import pandas as pd
from PIL import Image

from model_loader import load_models
from utils import preprocess, interpret_prediction
from config import CLASS_NAMES

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.title-box {
    background: linear-gradient(90deg, #1f77b4, #6a5acd);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 34px;
    font-weight: bold;
}
.subtitle {
    text-align:center;
    color:#cfcfcf;
    margin-bottom:25px;
}
.card {
    background-color:#1c1f26;
    padding:18px;
    border-radius:14px;
    border:1px solid #2f3440;
}
.small-text {
    font-size:14px;
    color:#cccccc;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title-box">Skin Cancer Detection using Deep Learning</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a dermoscopic image and compare predictions from multiple CNN models</p>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("About Project")
st.sidebar.info("""
This project predicts skin lesion classes using:

- ResNet50  
- EfficientNetB0  

Dataset used: **HAM10000**
""")

st.sidebar.warning("Educational tool only. Not a medical diagnosis.")

# ---------------- CLASS INFO ----------------
class_meanings = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesions"
}

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Skin Lesion Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### Model Predictions")

        with st.spinner("Loading models and predicting..."):
            models = load_models()

            results = []

            for model_name, model in models.items():

                input_img = preprocess(image, model_name)

                preds = model.predict(input_img, verbose=0)[0]

                class_name, confidence, label = interpret_prediction([preds])

                results.append({
                    "Model": model_name,
                    "Predicted Class": class_name.upper(),
                    "Confidence %": round(confidence * 100, 2),
                    "Status": label
                })

                st.markdown(f"#### {model_name}")

                if label == "Suspicious":
                    st.error(f"⚠️ Suspicious ({class_name.upper()})")
                else:
                    st.success(f"✅ Not Suspicious ({class_name.upper()})")

                st.progress(float(confidence))
                st.write(f"Confidence: **{confidence*100:.2f}%**")

                st.markdown("Top Class Probabilities:")

                prob_dict = {
                    CLASS_NAMES[i].upper(): float(preds[i])
                    for i in range(len(CLASS_NAMES))
                }

                st.bar_chart(prob_dict)

                st.markdown("---")

# ---------------- COMPARISON TABLE ----------------
        st.subheader("Model Comparison")

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

# ---------------- CLASS GUIDE ----------------
st.markdown("## Skin Lesion Class Meanings")

guide_df = pd.DataFrame({
    "Class Code": list(class_meanings.keys()),
    "Meaning": list(class_meanings.values())
})

st.dataframe(guide_df, use_container_width=True)

# ---------------- DATASET INFO ----------------
st.markdown("## Dataset Description")

st.markdown("""
**HAM10000** is a benchmark dermoscopic image dataset containing over **10,000 skin lesion images** across multiple diagnostic categories.

It is widely used for training and evaluating AI models for skin lesion classification.
""")

# ---------------- ABOUT ----------------
st.markdown("## About This Application")

st.markdown("""
This web application was developed as a final year engineering project.

### Technologies Used:
- Python  
- TensorFlow / Keras  
- Streamlit  
- CNN Architectures:
    - ResNet50
    - EfficientNetB0

### Purpose:
To demonstrate automated skin lesion classification using deep learning.
""")

# ---------------- DISCLAIMER ----------------
st.warning("This tool is for educational and research purposes only. It does not replace professional medical diagnosis.")