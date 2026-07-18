"""
utils.py
Helper functions for loading the trained EfficientNetB0 model, the
class-index mapping, preprocessing camera/uploaded images, and running
predictions with a confidence-based out-of-distribution guard.
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = os.path.join("model", "eff_model.keras")
CLASS_INDICES_PATH = os.path.join("model", "class_indices.json")
IMG_SIZE = (150, 150)  # must match the input size used during training

# Below this confidence, we tell the user we're not sure rather than
# confidently guessing. Tune this after looking at real-world photos.
CONFIDENCE_THRESHOLD = 0.55


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the fine-tuned EfficientNetB0 model once and cache it across reruns."""
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_class_indices():
    """
    Load class_indices.json (as saved from train_generator.class_indices,
    i.e. {"Tomato": 0, "Potato": 1, ...}) and invert it to {0: "Tomato", ...}
    so it can be used directly to map a predicted index back to a label.
    """
    with open(CLASS_INDICES_PATH, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Prepare a PIL image (from st.camera_input or file_uploader) for
    EfficientNetB0 inference:
      - fix EXIF rotation (phone photos are frequently mis-oriented)
      - convert to RGB (camera photos are sometimes RGBA/L)
      - resize to the training input size
      - apply EfficientNet's own preprocess_input (NOT manual /255. scaling)
    """
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(IMG_SIZE)

    img_array = np.array(pil_image).astype("float32")
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_top_k(model, img_array: np.ndarray, idx_to_class: dict, k: int = 3):
    """
    Run inference and return a list of (label, probability) tuples,
    sorted descending, for the top-k classes.
    """
    probs = model.predict(img_array, verbose=0)[0]
    top_k_idx = np.argsort(probs)[::-1][:k]
    return [(idx_to_class[i], float(probs[i])) for i in top_k_idx]


def is_confident(top_predictions) -> bool:
    """Out-of-distribution guard: reject low-confidence top-1 predictions."""
    if not top_predictions:
        return False
    return top_predictions[0][1] >= CONFIDENCE_THRESHOLD
