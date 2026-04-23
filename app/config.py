import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = {
    "ResNet50": os.path.join(BASE_DIR, "models", "ResNet50_model.keras"),
    "EfficientNetB0": os.path.join(BASE_DIR, "models", "efficientNet_model.keras"),
}

IMG_SIZE = (224, 224)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

SUSPICIOUS_CLASSES = ["mel", "bcc", "akiec"]