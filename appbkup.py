import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Class labels based on CIFAR-10
class_labels = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load model
@st.cache_resource
def load_cnn_model():
    return load_model("keras_cifar10_trained_model.h5")

model = load_cnn_model()

st.title("🔍 Low-Resolution Image Classifier")
st.write("Upload any image and it will be classified into one of the 10 categories.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize to 32x32 like training
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    st.success(f"### 🧠 Prediction: {class_labels[predicted_class]}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
