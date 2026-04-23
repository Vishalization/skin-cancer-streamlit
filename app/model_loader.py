import tensorflow as tf
import streamlit as st
from config import MODEL_PATHS


@st.cache_resource
def load_models():
    models = {}

    for name, path in MODEL_PATHS.items():
        models[name] = tf.keras.models.load_model(
            path,
            compile=False
        )

    return models