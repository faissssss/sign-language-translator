import streamlit as st
import cv2
import numpy as np
import sys
import os

# Add src to path to import RealTimeDetector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.realtime_detector import RealTimeDetector

st.set_page_config(page_title="ISL Translator", page_icon="👋")

# Hide the chat input and center title
st.markdown("""
    <style>
        /* Hide chat input */
        .stChatInput, .stChatFloatingInputContainer {
            display: none !important;
        }
        /* Center title and subtitle */
        h1, .stMarkdown p {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("👋 ISL Translator")
st.write("Translate Indonesian Sign Language gestures into text in real-time.")

# Cache the detector so it doesn't reload on every frame
@st.cache_resource
def load_detector():
    return RealTimeDetector()

try:
    detector = load_detector()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Camera input - uses browser webcam via Streamlit Cloud
st.subheader("📸 Capture a Sign")
img_file = st.camera_input("Show a sign gesture to the camera")

if img_file is not None:
    # Convert uploaded image to numpy array
    bytes_data = img_file.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)
    
    # Run prediction
    processed_frame, prediction, confidence = detector.predict(frame)
    
    # Convert BGR to RGB for display
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # Display results
    st.image(rgb_frame, caption="Processed Image", use_container_width=True)
    
    if prediction and confidence > 0.5:
        st.success(f"### Detected: **{prediction}** ({confidence:.1%} confidence)")
    elif prediction:
        st.warning(f"Low confidence: {prediction} ({confidence:.1%})")
    else:
        st.info("No hand detected. Please show your hand clearly in the frame.")
else:
    st.info("👆 Click the camera button above to capture a sign gesture.")
