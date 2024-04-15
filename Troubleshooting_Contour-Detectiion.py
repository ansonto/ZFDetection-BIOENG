import cv2
import numpy as np
import streamlit as st

st.title("Contour Extraction for Testing Code")

def extract_contours(gray_image, contour_area_threshold=100):
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, adaptive_threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(adaptive_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > contour_area_threshold:
            object_contours.append(contour)
    return object_contours

def visualize_contours(image, contours):
    contour_visualization = np.zeros_like(image)
    cv2.drawContours(contour_visualization, contours, -1, (255, 255, 255), -1)
    return contour_visualization

uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Please upload a maximum of 5 images.")
    else:
        for uploaded_file in uploaded_files:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contours = extract_contours(gray_image)
            contour_visualization = visualize_contours(image, contours)
            st.subheader(f"Image: {uploaded_file.name}")
            st.image(image, caption="Original Image", use_column_width=True)
            st.image(contour_visualization, caption="Contour Visualization", use_column_width=True)
else:
    st.warning("Please upload images to get started.")
