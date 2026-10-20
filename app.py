"""app.py - the web page (Streamlit). It only calls functions from engine.py."""
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import engine

st.title("ChromaSketch")
st.write("Turn your photo into a pencil sketch, color pencil drawing or cartoon.")

file = st.file_uploader(" Upload a photo", type=["jpg", "jpeg", "png"])

style = st.sidebar.radio("Choose style",
                         ["Pencil Sketch", "Color Pencil", "Cartoon"])

if style == "Cartoon":
    iterations = st.sidebar.slider("Smoothness (filter passes)", 1, 10, 5)
else:
    blur_size = st.sidebar.slider("Blur size", 3, 61, 21, step=2)
    if style == "Color Pencil":
        saturation = st.sidebar.slider("Color boost", 1.0, 3.0, 1.5, 0.1)

if file is None:
    st.info("Upload a photo above to begin.")
    st.stop()

img = cv2.cvtColor(np.array(Image.open(file).convert("RGB")), cv2.COLOR_RGB2BGR)
img = engine.resize_image(img)

if style == "Pencil Sketch":
    result = engine.pencil_sketch(img, blur_size)
elif style == "Color Pencil":
    result = engine.color_pencil(img, saturation, blur_size)
else:
    result = engine.cartoonify(img, iterations)

col1, col2 = st.columns(2)
col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
col2.image(result, caption=style, channels="GRAY" if result.ndim == 2 else "BGR")

ok, png = cv2.imencode(".png", result)
st.download_button("Download result", png.tobytes(), "result.png", "image/png")