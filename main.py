import streamlit as st

# ✅ This MUST be the first Streamlit command
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import gdown

# Step 1: Define model path and correct Google Drive file ID
MODEL_PATH = "model.h5"
FILE_ID = "1Xv3Lc89oDKNetFoNS_-pq43w7Ic3uj_z"  # <-- Correct ID

# Step 2: Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait."):
        download_url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(download_url, MODEL_PATH, quiet=False)

# Step 3: Load the model
@st.cache_resource
def load_brain_model():
    return load_model(MODEL_PATH)

model = load_brain_model()

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Step 4: Prediction function
def predict_tumor(image, model, image_size=128):
    image = image.resize((image_size, image_size))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    
    if class_labels[predicted_class_index] == 'notumor':
        result = "No Tumor"
    else:
        result = f"Tumor: {class_labels[predicted_class_index]}"
    
    return result, confidence_score

# Step 5: Streamlit UI
st.title("🧠 Brain Tumor Detection App")
st.write("Upload a brain MRI image to detect tumor type (if any).")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Predicting..."):
        result, confidence = predict_tumor(image, model)
    
    st.markdown(f"### 🔍 Result: {result}")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
