import streamlit as st
from camera_input_live import camera_input_live
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title="🔥 Real-Time Object Detection", layout="wide")
st.title("⚡ Instant Object Detection (Streamlit Cloud Optimized)")

model = YOLO("yolov8n.pt")

frame = camera_input_live()

if frame is not None:
    start = time.time()
    img = Image.open(frame).convert("RGB")
    np_img = np.array(img)
    results = model.predict(np_img, conf=0.01, verbose=False)  # show everything
    annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)  # fix blue tint
    end = time.time()
    fps = 1 / (end - start + 1e-6)

    c1, c2 = st.columns(2)
    with c1:
        st.image(np_img, caption="📸 Live Camera Feed", use_container_width=True)
    with c2:
        st.image(annotated, caption=f"🎯 Detections ({fps:.1f} FPS)", use_container_width=True)
        