import streamlit as st
from io import BytesIO
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

# --- Custom CSS & Theme ---
st.set_page_config(page_title="RL AI LOVAIC", page_icon="logo.png", layout="wide")

# Load brand logo
logo = Image.open("logo.png")
st.sidebar.image(logo, width=150)

# Hide default Streamlit style
st.markdown("""
  <style>
  /* Remove Streamlit header and footer */
  header, footer {visibility: hidden;}
  /* Main background and font colors */
  .reportview-container, .sidebar .sidebar-content {
    background: #1e1e2f;
    color: #ffffff;
  }
  /* Custom buttons */
  .stButton>button {
    background-color: #ff6f61;
    color: white;
    border-radius: 8px;
    padding: 0.4em 1.2em;
  }
  .stButton>button:hover {
    background-color: #ff876f;
  }
  /* Image styling */
  .uploaded-image {
    border: 3px solid #ff6f61;
    border-radius: 10px;
    margin: 0.5em;
  }
  </style>
""", unsafe_allow_html=True)

# Sidebar title
st.sidebar.title("RL AI - LOVAIC")

# Load model once
@st.cache_resource
def load_classifier():
    return load_model(os.path.join("keras_cifar10_trained_model.h5")) #return load_model(os.path.join("saved_models", "keras_cifar10_trained_model.h5"))

model = load_classifier()
class_labels = ['Airplane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

st.image("lovaic.png",width=200)
st.title("Upload Images for Classification")
st.write("Select one or more images and get predictions below.")

# Multi-image uploader
uploaded = st.file_uploader("Choose image files", type=["jpg","png","jpeg"], accept_multiple_files=True)
if uploaded:
    cols = st.columns(len(uploaded))
    for file, col in zip(uploaded, cols):
        img = Image.open(file).convert("RGB")
        col.image(img, use_column_width=True, caption=file.name, output_format="auto", 
                  clamp=False)

        # Preprocess
        img_resized = img.resize((32,32))
        arr = np.array(img_resized).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Predict
        pred = model.predict(arr)
        cls = class_labels[np.argmax(pred)]
        conf = np.max(pred) * 100

        col.markdown(f"**{cls}** — {conf:.1f}% confidence")

