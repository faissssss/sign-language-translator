import streamlit as st
import cv2
import numpy as np
import sys
import os

# Add src to path to import RealTimeDetector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.realtime_detector import RealTimeDetector

st.set_page_config(page_title="ISL Translator", page_icon="👋")

# Minimal CSS
st.markdown("""
    <style>
        h1 { text-align: center; }
        .stAlert { display: none; }
    </style>
""", unsafe_allow_html=True)

st.title("👋 ISL Translator")

# Cache the detector
@st.cache_resource
def load_detector():
    return RealTimeDetector()

detector = load_detector()

# Camera input
img_file = st.camera_input("", label_visibility="collapsed")

if img_file is not None:
    # Convert to numpy array
    bytes_data = img_file.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)
    
    # Run prediction
    processed_frame, prediction, confidence = detector.predict(frame)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # Display
    st.image(rgb_frame, use_container_width=True)
    
    if prediction and confidence > 0.5:
        st.success(f"**{prediction}** ({confidence:.0%})")
