import cv2
import numpy as np
import streamlit as st

st.title("Visualization for Testing Code")

def is_embryo_present(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    num_pixels = cv2.countNonZero(binary)
    return num_pixels > 1500

def process_image(image, min_fluorescence=200, max_fluorescence=255):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, adaptive_threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(adaptive_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    embryo_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            embryo_contours.append(contour)

    if len(embryo_contours) > 0:
        for contour in embryo_contours:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, f"fish", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            contour_region = image[y:y+h, x:x+w]

            hsv = cv2.cvtColor(contour_region, cv2.COLOR_BGR2HSV)

            lower_color = np.array([0, 0, min_fluorescence])
            upper_color = np.array([255, 255, max_fluorescence])

            binary = cv2.inRange(hsv, lower_color, upper_color)

            fluorescent_pixels = contour_region[binary > 0]
            if len(fluorescent_pixels) > 0:
                fluorescence_level = np.mean(fluorescent_pixels)
                cv2.putText(image, f"FL: {fluorescence_level:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(image, "No Fluorescence", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return image
    else:
        return image

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if is_embryo_present(image):
        st.write("Embryo Present")
        processed_image = process_image(image)
        st.image(processed_image, channels="BGR", use_column_width=True)
    else:
        st.write("No Embryo Detected")
else:
    st.warning("Please upload an image to get started.")
    