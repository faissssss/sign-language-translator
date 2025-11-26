import streamlit as st
import cv2
import numpy as np
import sys
import os

# Add src to path to import RealTimeDetector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.realtime_detector import RealTimeDetector

st.set_page_config(page_title="ISL Translator", page_icon="👋")

st.title("👋 ISL Translator")
st.write("Translate Indonesian Sign Language gestures into text in real-time.")

# Cache the detector so it doesn't reload on every frame
@st.cache_resource
def load_detector():
    return RealTimeDetector(model_path='models/isl_classifier.p')

try:
    detector = load_detector()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Webcam input
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])
TEXT_WINDOW = st.empty()

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to capture video")
        break
    
    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)
    
    # Run prediction
    # The detector draws on the frame and returns it
    processed_frame, prediction, confidence = detector.predict(frame)
    
    # Convert BGR to RGB for Streamlit
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # Update UI
    FRAME_WINDOW.image(rgb_frame)
    
    # Display text with styling
    if prediction != "Waiting...":
        color = "green" if confidence > 0.7 else "orange"
        TEXT_WINDOW.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                <h2 style="color: {color}; margin:0;">{prediction}</h2>
                <p style="margin:0;">Confidence: {confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        TEXT_WINDOW.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                <h2 style="color: gray; margin:0;">Waiting...</h2>
            </div>
            """, unsafe_allow_html=True)

else:
    camera.release()
    st.write("Camera stopped.")
