import streamlit as st
from PIL import Image
import numpy as np
import os
import joblib
import gdown
import io
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf

st.title("ASD DETECTION SYSTEM ")
st.write(
    " 🔍 Detecting occurence and severity of Autism Spectrum Disorder in children using image 👦👧."
)

# Model loader from Google Drive
@st.cache_resource
def load_model_from_drive(file_id, task_model_name):
    model_path = f"models/{task_model_name}.pkl"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    return model

@st.cache_resource
def load_feature_extractor(model_name):
    if "VGG16" in model_name:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        return base_model, preprocess_vgg
    elif "ResNet" in model_name:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        return base_model, preprocess_resnet
    elif "EfficientNet" in model_name:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        return base_model, preprocess_efficient
    else:
        raise ValueError("Unknown model type. Please include 'VGG16', 'ResNet', or 'EfficientNet' in the model name.")

def preprocess_image(img, model_name):
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    base_model, preprocess_fn = load_feature_extractor(model_name)
    img_array = preprocess_fn(img_array)
    features = base_model.predict(img_array)
    features = features.flatten().reshape(1, -1)
    return features


# Prediction logic
def predict(img, model, model_name):
    features = preprocess_image(img, model_name)
    prediction = model.predict(features)[0]

# Predict confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = np.max(proba) * 100  # max confidence
    else:
        confidence = None  # fallback if not available

    label_map = {
        0: "Non-ASD",
        1: "ASD Mild",
        2: "ASD Moderate",
        3: "ASD Severe"
    }
    label = label_map.get(prediction, prediction)

    if confidence is not None:
        return f"Predicted as: {label} (Confidence: {confidence:.2f}%)"
    else:
        return f"Predicted as: {label}"
    
    return f"Predicted as: {label}"

# Sidebar navigation
st.sidebar.title("ASD App")
page = st.sidebar.radio("Choose Task", ["Coloring 🎨", "Drawing 🖍️", "Handwriting ✍🏻"])

# Define models for each task with their Google Drive IDs
model_drive_links = {
    "Coloring 🎨": {
        "EfficientNet": "1plL7WQF4W6B5YUmqX6rWDrlkNlcM_iH9", 
        "ResNet": "147KE-MDukcFqaKE9v5bDk1xhUUXS5WBn",
        "VGG16": "1zIwLe_x0bJuLCPLO4YZlhgL84Mkt7wYi"
    },
    "Drawing 🖍️": {
        "EfficientNet": "1H5Qxtq8HfWd4arXPgCMC-CjLZEo-Y3Cg",
        "ResNet": "1eFHGMvwwk8Oc446e-6dHA2JdODieVUQY",
        "VGG16": "1BmTfG1epP132zkP41NmzbZ-Hz_utrXZy"
    },
    "Handwriting ✍🏻": {
        "EfficientNet": "17laZFA2sF4JVtByBHePA-UX1HOm_HtTL",
        "ResNet": "1Rwl4XwFYBXv8M41BHRbxv4ACo_Hi3K_i",
        "VGG16": "1ZDaUIZmvIqcufYFaSEHnASqZFGDbX-dL"
    }
}

def show_page(task):
    st.title(f"{task}")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", use_container_width=True)

        model_name = st.selectbox("Select Model", list(model_drive_links[task].keys()))
        model_id = model_drive_links[task][model_name]

        if st.button("Predict"):
            with st.spinner("Loading model and predicting..."):
                model = load_model_from_drive(model_id, model_name)
                result = predict(image, model, model_name)
                st.success(result)

# Routing
if page == "Coloring 🎨":
    show_page(page)
elif page == "Drawing 🖍️":
    show_page(page)
elif page == "Handwriting ✍🏻":
    show_page(page)
