# app.py
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np

st.set_page_config(page_title="YOLOv8 Real-time Detection", layout="wide")
st.title("Real-time Object Detection — YOLOv8 (Streamlit + webrtc)")

# Sidebar controls
conf_thresh = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
model_size = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
imgsz = st.sidebar.slider("Image size (px)", 320, 1280, 640, step=64)

st.sidebar.markdown("**Tips:** Use `yolov8n` for faster inference. Use GPU if available.")

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model = load_model(model_size)

class YoloTransformer(VideoTransformerBase):
    def __init__(self):
        # Use the cached model instance
        self.model = model

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        # Run inference (single image)
        results = self.model.predict(img, conf=conf_thresh, imgsz=imgsz, verbose=False)
        # Annotate and return
        annotated = results[0].plot()  # BGR image with boxes
        return annotated

webrtc_streamer(key="yolov8", video_transformer_factory=YoloTransformer,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False})
