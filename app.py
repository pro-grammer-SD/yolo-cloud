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
    img = Image.open(frame)
    img = np.array(img)
    results = model.predict(img, conf=0.5, verbose=False)
    annotated = results[0].plot()
    st.image(annotated, channels="BGR", use_container_width=True)
    