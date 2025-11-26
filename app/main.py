from flask import Flask, render_template, Response
from flask_cors import CORS
import cv2
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.realtime_detector import RealTimeDetector

app = Flask(__name__)
CORS(app)

# Initialize Detector
# Use absolute path for model to avoid directory issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'isl_classifier.p')
detector = RealTimeDetector(model_path=MODEL_PATH)

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame (mirror effect) - same as training
        frame = cv2.flip(frame, 1)
        
        # Run prediction (draws on frame)
        frame, prediction, confidence = detector.predict(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    finally:
        camera.release()
