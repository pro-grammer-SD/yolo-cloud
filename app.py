import streamlit as st
from camera_input_live import camera_input_live
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Live Object Detection", layout="wide")
st.title("🎯 Live Object Detection")

model = YOLO("yolov8n.pt")

frame = camera_input_live()

if frame is not None:
    img = Image.open(frame).convert("RGB")
    np_img = np.array(img)
    st.image(np_img, caption="📸 Live Camera Feed", use_container_width=True)

    results = model.predict(np_img, conf=0.5, verbose=False)
    annotated = results[0].plot()
    st.image(annotated, channels="BGR", caption="🎯 Detected Objects", use_container_width=True)
    