import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

from settings import DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE


st.title("Crack Detections with Yolo V5s")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
confidence_threshold = st.slider(
    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)