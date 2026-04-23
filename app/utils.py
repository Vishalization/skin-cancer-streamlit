import numpy as np
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess

from config import IMG_SIZE, CLASS_NAMES, SUSPICIOUS_CLASSES


def preprocess(img, model_name):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if model_name == "ResNet50":
        return resnet_preprocess(img_array)

    elif model_name == "EfficientNetB0":
        return efficient_preprocess(img_array)

    else:
        raise ValueError("Unknown model")


def interpret_prediction(preds):
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    class_name = CLASS_NAMES[class_idx]

    label = "Suspicious" if class_name in SUSPICIOUS_CLASSES else "Not Suspicious"

    return class_name, confidence, label