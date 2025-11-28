from flask import Flask, render_template, Response
from flask_cors import CORS
import cv2
import os
import sys
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.realtime_detector import RealTimeDetector

app = Flask(__name__)
CORS(app)

# Initialize Detector
detector = RealTimeDetector()

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        try:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera. Retrying...")
                time.sleep(0.1)
                continue
            
            # Flip frame (mirror effect) - same as training
            frame = cv2.flip(frame, 1)
            
            # Run prediction (draws on frame)
            try:
                frame, prediction, confidence = detector.predict(frame)
            except Exception as e:
                print(f"Prediction error: {e}")
                # Continue displaying frame even if prediction fails
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5001)
    finally:
        camera.release()
