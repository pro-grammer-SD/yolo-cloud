import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2

st.set_page_config(page_title="Live Object Detector", layout="wide")
st.title("🎯 Live Hitbox Object Detector")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

class ObjectDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="object-detection",
    video_transformer_factory=ObjectDetector,
    media_stream_constraints={"video": True, "audio": False},
)
