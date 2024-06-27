import cv2
import numpy as np
import streamlit as st

st.title("Zebrafish Embryo Fluorescence Detector")

def is_embryo_present(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)
    num_pixels = cv2.countNonZero(binary)
    return num_pixels > 1500

def process_frame(frame, min_fluorescence=200, max_fluorescence=255):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for GFP fluorescence in HSV
    lower_hue = np.array([35, 50, 50])
    upper_hue = np.array([85, 255, 255])

    # Create a mask to isolate regions with GFP fluorescence
    mask = cv2.inRange(hsv_image, lower_hue, upper_hue)

    # Apply mask to get only the fluorescent regions
    fluorescent_regions = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the isolated regions to grayscale
    gray_fluorescent = cv2.cvtColor(fluorescent_regions, cv2.COLOR_BGR2GRAY)

    # Calculate the average intensity of the fluorescent pixels
    fluorescent_pixels = gray_fluorescent[gray_fluorescent > 0]
    if len(fluorescent_pixels) > 0:
        fluorescence_level = np.mean(fluorescent_pixels)
    else:
        fluorescence_level = 0

    return frame, fluorescence_level

def draw_zebrafish_contour(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [largest_contour], -1, (255, 255, 255), 2)
        
    return frame

# Choose video source and capture a frame
video_source = st.radio("Select Video Source", ("Webcam", "Upload"))

if video_source == "Webcam":
    video_capture = cv2.VideoCapture(0)
else:
    # Video upload
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getvalue())
        # Read video from uploaded file
        video_capture = cv2.VideoCapture("temp_video.mp4")

if 'video_capture' in locals() and video_capture.isOpened():
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Process one frame per second
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            if is_embryo_present(frame):
                st.write("Embryo Present")
                frame_with_contour = draw_zebrafish_contour(frame)
                processed_frame, fluorescence_level = process_frame(frame_with_contour)
                st.image(processed_frame, channels="BGR", use_column_width=True)
                st.write(f"Average Fluorescence Level: {fluorescence_level:.2f}")
            else:
                st.write("No Embryo Detected")

        frame_count += 1

    # Release the video capture
    video_capture.release()
else:
    st.write("Failed to capture frame from video source.")
