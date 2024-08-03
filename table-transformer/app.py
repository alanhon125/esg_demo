from datetime import datetime

import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(layout="wide")


st.sidebar.header("TABLE RECOGNITION DEMO")


uploaded_file = st.file_uploader(label="UPLOAD FILE", type=["png", "jpg", "jpeg"])

from utils import add_margin, resize_img

image = None
if uploaded_file:
    # image = Image.open(uploaded_file).convert("RGB")
    image = Image.open(uploaded_file).convert("RGB")
    st.text(f"{image.size[0]}, {image.size[1]}")
    image = add_margin(image, 50, 50, 50, 50, (255,255,255,255))
    st.text(f"{image.size[0]}, {image.size[1]}")
    image, _ = resize_img(image, 800, 800)
    st.text(f"{image.size[0]}, {image.size[1]}")


from core import TableRecognizer, TableDetector


m = TableRecognizer(
    checkpoint_path="/home/data/lqy/model/model_20.pth"
)

m2 = TableDetector(
    checkpoint_path="/home/data/lqy/pubtables1m_detection_detr_r18.pth"
)


def main():
    global image
    if image is None:
        return

    placeholder = st.image(image, width=500)  # display image

    with st.spinner("🤖 AI is at Work! "):
        start_time = datetime.now()
        results = m2.predict(image_path=image)

        output_image = results["debug_image"]
        consume_time = datetime.now() - start_time

        st.text(f"Consume time: {consume_time}")

        placeholder.empty()
        st.image([image, output_image], caption=["input", "output"], width=500)
        st.text(str(results))


if __name__ == "__main__":
    main()
