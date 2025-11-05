import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(page_title="YOLO Object Detector", layout="centered")

model = YOLO("yolov8n.pt")

st.title("🧠 Smart Object Identifier")

img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    results = model.predict(np.array(img), conf=0.1, verbose=False)
    res = results[0]
    boxes = res.boxes
    names = res.names
    labels = [names[int(i)] for i in boxes.cls]
    confs = [round(float(c) * 100, 2) for c in boxes.conf]
    df = pd.DataFrame({"Item": labels, "Confidence (%)": confs})
    df = df.groupby("Item").agg({"Confidence (%)": "mean",}).reset_index()
    df["Count"] = df["Item"].map(lambda x: f"x{labels.count(x)}" if labels.count(x) > 1 else "")
    st.image(res.plot(), use_container_width=True)
    st.dataframe(df, use_container_width=True)
    