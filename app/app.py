import streamlit as st
import cv2
import numpy as np
import sys
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Add src to path to import RealTimeDetector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.realtime_detector import RealTimeDetector

st.set_page_config(page_title="ISL Translator", page_icon="👋", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        /* Hide chat input */
        .stChatInput, .stChatFloatingInputContainer {
            display: none !important;
        }
        /* Center title and subtitle */
        h1 { text-align: center; }
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("👋 ISL Translator")
st.write("Translate Indonesian Sign Language gestures into text in real-time.")

# RTC Configuration for STUN server (needed for WebRTC)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Cache the detector
@st.cache_resource
def load_detector():
    return RealTimeDetector()

# Global variable for current prediction (for display outside video)
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = ""
    st.session_state.current_confidence = 0.0

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = load_detector()
        self.prediction = ""
        self.confidence = 0.0
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip for selfie view
        img = cv2.flip(img, 1)
        
        # Run prediction
        processed_frame, prediction, confidence = self.detector.predict(img)
        
        # Store prediction
        if prediction:
            self.prediction = prediction
            self.confidence = confidence
        
        # Draw prediction on frame
        if prediction and confidence > 0.5:
            cv2.putText(processed_frame, f"{prediction} ({confidence:.0%})", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

try:
    detector = load_detector()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Instructions
st.info("📹 **Instructions:** Click 'START' below to begin real-time detection. Show sign gestures to your camera!")

# WebRTC Streamer
ctx = webrtc_streamer(
    key="sign-language-detector",
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Show current detection status
if ctx.video_processor:
    st.markdown("### 🔍 Detection Active")
    st.write("The detected sign will appear on the video feed in real-time.")
else:
    st.warning("👆 Click **START** above to begin real-time sign language detection.")
