import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("Image Statistics for Testing Code")

def is_embryo_present(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    num_pixels = cv2.countNonZero(binary)
    return num_pixels > 1500

def process_image(image, min_fluorescence=200, max_fluorescence=255):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, adaptive_threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = _
    contours, _ = cv2.findContours(adaptive_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    embryo_contours = []
    fluorescent_pixel_intensities = []
    non_fluorescent_pixel_intensities = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            embryo_contours.append(contour)

    if len(embryo_contours) > 0:
        for contour in embryo_contours:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            contour_region = image[y:y+h, x:x+w]
            hsv = cv2.cvtColor(contour_region, cv2.COLOR_BGR2HSV)
            lower_color = np.array([0, 0, min_fluorescence])
            upper_color = np.array([255, 255, max_fluorescence])
            binary = cv2.inRange(hsv, lower_color, upper_color)
            fluorescent_pixels = contour_region[binary > 0]
            non_fluorescent_pixels = contour_region[binary == 0]
            fluorescent_pixel_intensities.extend(fluorescent_pixels.flatten())
            non_fluorescent_pixel_intensities.extend(non_fluorescent_pixels.flatten())

            if len(fluorescent_pixels) > 0:
                fluorescence_level = np.mean(fluorescent_pixels)
                cv2.putText(image, f"FL: {fluorescence_level:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(image, "No Fluorescence", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return image, fluorescent_pixel_intensities, non_fluorescent_pixel_intensities, threshold_value

uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Please upload a maximum of 5 images.")
    else:
        fluorescent_pixel_counts = []
        non_fluorescent_pixel_counts = []
        fluorescent_pixel_intensities_all = []
        non_fluorescent_pixel_intensities_all = []

        for uploaded_file in uploaded_files:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            st.subheader(f"Image: {uploaded_file.name}")
            
            if is_embryo_present(image):
                st.write("Embryo Present")
                processed_image, fluorescent_pixel_intensities, non_fluorescent_pixel_intensities, threshold_value = process_image(image)
                st.image(processed_image, channels="BGR", use_column_width=True)

                avg_fluorescent_intensity = np.mean(fluorescent_pixel_intensities)
                avg_non_fluorescent_intensity = np.mean(non_fluorescent_pixel_intensities)
                
                st.write(f"Avg. Fluorescent Intensity: {avg_fluorescent_intensity:.2f}")
                st.write(f"Avg. Non-Fluorescent Intensity: {avg_non_fluorescent_intensity:.2f}")
                st.write(f"Threshold Value: {threshold_value}")

                fluorescent_pixel_counts.append(len(fluorescent_pixel_intensities))
                non_fluorescent_pixel_counts.append(len(non_fluorescent_pixel_intensities))
                fluorescent_pixel_intensities_all.extend(fluorescent_pixel_intensities)
                non_fluorescent_pixel_intensities_all.extend(non_fluorescent_pixel_intensities)

                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

                fig, ax = plt.subplots()
                ax.plot(histogram)
                ax.axvline(threshold_value, color='red', linestyle='dashed', linewidth=2, label='Otsu Threshold')
                ax.set_title('Grayscale Histogram')
                ax.set_xlabel('Pixel Intensity')
                ax.set_ylabel('Frequency')
                ax.legend()
                st.pyplot(fig)
            else:
                st.write("No Embryo Detected")

        if len(fluorescent_pixel_counts) > 0:
            avg_fluorescent_pixels = np.mean(fluorescent_pixel_counts)
            avg_non_fluorescent_pixels = np.mean(non_fluorescent_pixel_counts)

            fluorescent_std = np.std(fluorescent_pixel_counts)
            non_fluorescent_std = np.std(non_fluorescent_pixel_counts)

            fig, ax = plt.subplots()
            categories = ['Non-Fluorescent', 'Fluorescent']
            values = [avg_non_fluorescent_pixels, avg_fluorescent_pixels]
            errors = [non_fluorescent_std, fluorescent_std]
            bar_colors = ['#8da0cb', '#66c2a5'] 
            ax.bar(categories, values, yerr=errors, capsize=10, color=bar_colors)
            ax.set_ylabel('Average Number of Pixels')
            ax.set_title('Average Number of Non-Fluorescent and Fluorescent Pixels')
            fig.tight_layout()
            st.pyplot(fig)

            avg_fluorescent_intensity = np.mean(fluorescent_pixel_intensities_all)
            avg_non_fluorescent_intensity = np.mean(non_fluorescent_pixel_intensities_all)

            fluorescent_intensity_std = np.std(fluorescent_pixel_intensities_all)
            non_fluorescent_intensity_std = np.std(non_fluorescent_pixel_intensities_all)

            fig, ax = plt.subplots()
            categories = ['Non-Fluorescent', 'Fluorescent']
            values = [avg_non_fluorescent_intensity, avg_fluorescent_intensity]
            errors = [non_fluorescent_intensity_std, fluorescent_intensity_std]
            bar_colors = ['#8da0cb', '#66c2a5'] 
            ax.bar(categories, values, yerr=errors, capsize=10, color=bar_colors)
            ax.set_ylabel('Average Pixel Intensity')
            ax.set_title('Average Intensity of Non-Fluorescent and Fluorescent Pixels')
            fig.tight_layout()
            st.pyplot(fig)
else:
    st.warning("Please upload images to get started.")
