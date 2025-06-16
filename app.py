<<<<<<< HEAD
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
  header, footer {visibility: hidden;}
  .reportview-container, .sidebar .sidebar-content {
    background: #1e1e2f;
    color: #ffffff;
  }
  .stButton>button {
    background-color: #ff6f61;
    color: white;
    border-radius: 8px;
    padding: 0.4em 1.2em;
  }
  .stButton>button:hover {
    background-color: #ff876f;
  }
  .uploaded-image {
    border: 3px solid #ff6f61;
    border-radius: 10px;
    margin: 0.5em;
  }
  </style>
""", unsafe_allow_html=True)

# Sidebar title and menu
st.sidebar.title("RL AI - LOVAIC")
task = st.sidebar.radio("📌 Select Task", ["Classify Images", "New Training"])

# Load model based on current session
@st.cache_resource
def load_classifier(path):
    return load_model(path)

default_model_path = "RLAI_LOVAIC_trained_model_1.h5"
model_path = st.session_state.get('model_path', default_model_path)
model = load_classifier(model_path)

class_labels = ['Airplane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

# ---------------- STAGE 1: Classify Images ----------------
if task == "Classify Images":
    st.image("lovaic.png", width=200)
    st.title("Upload Images for Classification")
    st.write("Select one or more images and get predictions below.")

    uploaded = st.file_uploader("Choose image files", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if uploaded:
        cols = st.columns(len(uploaded))
        for file, col in zip(uploaded, cols):
            img = Image.open(file).convert("RGB")
            col.image(img, use_column_width=True, caption=file.name, output_format="auto")

            img_resized = img.resize((32,32))
            arr = np.array(img_resized).astype("float32") / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr)
            cls = class_labels[np.argmax(pred)]
            conf = np.max(pred) * 100
            col.markdown(f"**{cls}** — {conf:.1f}% confidence")

# ---------------- STAGE 2: New Training ----------------
elif task == "New Training":
    st.title("🚀 Train a New Model on Your Dataset")

    dataset = st.file_uploader("📦 Upload ZIP of images (with subfolders as class names)", type=["zip"])
    epochs = st.sidebar.number_input("🔁 Number of Epochs", 1, 100, value=5)

    if dataset:
        with st.spinner("Processing and training..."):
            import zipfile, uuid, shutil
            import tensorflow as tf
            import matplotlib.pyplot as plt

            temp_dir = f"temp_train_{uuid.uuid4().hex}"
            os.makedirs(temp_dir, exist_ok=True)

            with zipfile.ZipFile(dataset, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                temp_dir, image_size=(32, 32), batch_size=32)

            class_names = dataset.class_names
            num_classes = len(class_names)

            new_model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255, input_shape=(32,32,3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])

            new_model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

            history = new_model.fit(dataset, epochs=epochs)

            # Accuracy/Loss plot
            acc = history.history['accuracy']
            loss = history.history['loss']
            epochs_range = range(1, epochs + 1)

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(epochs_range, acc, label='Accuracy', color='skyblue')
            ax[0].set_title('Training Accuracy')
            ax[1].plot(epochs_range, loss, label='Loss', color='salmon')
            ax[1].set_title('Training Loss')
            st.pyplot(fig)

            # Save and update model
            new_model_path = "saved_models/latest_model.h5"
            os.makedirs("saved_models", exist_ok=True)
            new_model.save(new_model_path)
            st.session_state['model_path'] = new_model_path
            load_classifier.clear()  # Refresh cache

            st.success(f"✅ New model trained and loaded from `{new_model_path}`.")
            shutil.rmtree(temp_dir)
=======
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
  header, footer {visibility: hidden;}
  .reportview-container, .sidebar .sidebar-content {
    background: #1e1e2f;
    color: #ffffff;
  }
  .stButton>button {
    background-color: #ff6f61;
    color: white;
    border-radius: 8px;
    padding: 0.4em 1.2em;
  }
  .stButton>button:hover {
    background-color: #ff876f;
  }
  .uploaded-image {
    border: 3px solid #ff6f61;
    border-radius: 10px;
    margin: 0.5em;
  }
  </style>
""", unsafe_allow_html=True)

# Sidebar title and menu
st.sidebar.title("RL AI - LOVAIC")
task = st.sidebar.radio("📌 Select Task", ["Classify Images", "New Training"])

# Load model based on current session
@st.cache_resource
def load_classifier(path):
    return load_model(path)

default_model_path = "RLAI_LOVAIC_trained_model_1.h5"
model_path = st.session_state.get('model_path', default_model_path)
model = load_classifier(model_path)

class_labels = ['Airplane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

# ---------------- STAGE 1: Classify Images ----------------
if task == "Classify Images":
    st.image("lovaic.png", width=200)
    st.title("Upload Images for Classification")
    st.write("Select one or more images and get predictions below.")

    uploaded = st.file_uploader("Choose image files", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if uploaded:
        cols = st.columns(len(uploaded))
        for file, col in zip(uploaded, cols):
            img = Image.open(file).convert("RGB")
            col.image(img, use_column_width=True, caption=file.name, output_format="auto")

            img_resized = img.resize((32,32))
            arr = np.array(img_resized).astype("float32") / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr)
            cls = class_labels[np.argmax(pred)]
            conf = np.max(pred) * 100
            col.markdown(f"**{cls}** — {conf:.1f}% confidence")

# ---------------- STAGE 2: New Training ----------------
elif task == "New Training":
    st.title("🚀 Train a New Model on Your Dataset")

    dataset = st.file_uploader("📦 Upload ZIP of images (with subfolders as class names)", type=["zip"])
    epochs = st.sidebar.number_input("🔁 Number of Epochs", 1, 100, value=5)

    if dataset:
        with st.spinner("Processing and training..."):
            import zipfile, uuid, shutil
            import tensorflow as tf
            import matplotlib.pyplot as plt

            temp_dir = f"temp_train_{uuid.uuid4().hex}"
            os.makedirs(temp_dir, exist_ok=True)

            with zipfile.ZipFile(dataset, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                temp_dir, image_size=(32, 32), batch_size=32)

            class_names = dataset.class_names
            num_classes = len(class_names)

            new_model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255, input_shape=(32,32,3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])

            new_model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

            history = new_model.fit(dataset, epochs=epochs)

            # Accuracy/Loss plot
            acc = history.history['accuracy']
            loss = history.history['loss']
            epochs_range = range(1, epochs + 1)

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(epochs_range, acc, label='Accuracy', color='skyblue')
            ax[0].set_title('Training Accuracy')
            ax[1].plot(epochs_range, loss, label='Loss', color='salmon')
            ax[1].set_title('Training Loss')
            st.pyplot(fig)

            # Save and update model
            new_model_path = "saved_models/latest_model.h5"
            os.makedirs("saved_models", exist_ok=True)
            new_model.save(new_model_path)
            st.session_state['model_path'] = new_model_path
            load_classifier.clear()  # Refresh cache

            st.success(f"✅ New model trained and loaded from `{new_model_path}`.")
            shutil.rmtree(temp_dir)
>>>>>>> bea0651 (Updated code for retrained model and deployment)
