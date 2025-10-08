import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from PIL import Image
import matplotlib.pyplot as plt

# =======================================================
# 🌤 Streamlit Page Configuration + Light Theme Styling
# =======================================================
st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

st.markdown("""
    <style>
    /* 🌈 Background and main content */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f9fafc 0%, #eef3f7 100%);
        color: #000000;
    }

    /* 💠 Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #0078D7;
        margin-bottom: 1.5rem;
    }

    /* 🌼 Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f2f5f9;
        color: #000000;
        border-right: 1px solid #dce3eb;
    }

    /* 🪄 Card styling */
    .result-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
    }

    /* 📊 Matplotlib figure background fix */
    .stPlotlyChart, .stPyplotChart {
        background-color: transparent !important;
    }

    /* 📤 Upload section header */
    .upload-header {
        color: #0078D7;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>😊 Facial Emotion Recognition — FER2013 Model</h1>", unsafe_allow_html=True)

# =======================================================
# ⚙️ Model Paths
# =======================================================
MODEL_DIR = r"D:\Streamlit\fer2013_image_model"
MODEL_H5 = r"D:\Streamlit\fer2013_image_model.h5"

# =======================================================
# 🔁 Convert SavedModel → .h5 (if needed)
# =======================================================
def convert_model_to_h5(model_dir, save_path):
    try:
        st.sidebar.info("⏳ Converting TensorFlow SavedModel to .h5 format...")
        model = load_model(model_dir)
        model.save(save_path)
        st.sidebar.success("✅ Model converted and saved successfully!")
        return save_path
    except Exception as e:
        st.sidebar.error(f"⚠️ Conversion failed: {e}")
        return None

# =======================================================
# 🚀 Load Model
# =======================================================
@st.cache_resource
def load_fer_model():
    if os.path.exists(MODEL_H5):
        model = load_model(MODEL_H5)
        st.sidebar.success("✅ Model (.h5) loaded successfully!")
        return model
    elif os.path.isdir(MODEL_DIR):
        converted = convert_model_to_h5(MODEL_DIR, MODEL_H5)
        if converted:
            model = load_model(converted)
            st.sidebar.success("✅ Model converted and loaded successfully!")
            return model
        else:
            st.sidebar.error("❌ Model conversion failed.")
            return None
    else:
        st.sidebar.error("❌ Model file not found.")
        return None

model = load_fer_model()

# =======================================================
# 😃 Emotion Labels
# =======================================================
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =======================================================
# 📸 Image Upload
# =======================================================
st.markdown("<div class='upload-header'>📸 Upload a Face Image</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # =======================================================
    # 👁️ Face Detection
    # =======================================================
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("⚠️ No face detected. Try another image.")
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=(0, -1))

            prediction = model.predict(roi_gray)
            emotion = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Draw detection box and label
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 150, 255), 2)
            cv2.putText(img_array, f"{emotion} ({confidence*100:.1f}%)",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

        with col2:
            st.image(img_array, caption="🎯 Detected Emotion", use_container_width=True)

        # =======================================================
        # 🎨 Result Card + Probability Chart
        # =======================================================
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0078D7;'>🧩 Detected Emotion: <span style='color:#333;'>{emotion}</span></h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#666;'>Confidence: {confidence*100:.2f}%</h4>", unsafe_allow_html=True)

        # Plot probability bar chart
        fig, ax = plt.subplots(figsize=(5,3))
        ax.barh(emotion_labels, prediction[0], color='#0078D7')
        ax.set_xlabel('Probability', color='#000')
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#ffffff')
        ax.tick_params(colors='#000')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

elif uploaded_file is None:
    st.info("📤 Please upload an image to analyze.")
elif model is None:
    st.error("❌ Model could not be loaded. Please check model path or format.")
