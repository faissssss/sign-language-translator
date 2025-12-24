import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

class RealTimeDetector:
    def __init__(self, model_path=None):
        print("Initializing RealTimeDetector...")
        # Calculate absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        
        if model_path:
            classifier_path = model_path if os.path.isabs(model_path) else os.path.join(project_dir, model_path)
        else:
            classifier_path = os.path.join(project_dir, 'models', 'isl_classifier.p')
        
        self.hand_landmarker_path = os.path.join(project_dir, 'models', 'hand_landmarker.task')
        
        # Load classifier model
        try:
            print(f"Loading classifier from {classifier_path}...")
            self.model_dict = pickle.load(open(classifier_path, 'rb'))
            self.model = self.model_dict['model']
            print("Classifier loaded successfully.")
        except FileNotFoundError:
            print(f"Classifier not found at {classifier_path}. Please train the model first.")
            self.model = None
        except Exception as e:
            print(f"Error loading classifier: {e}")
            self.model = None
        
        # Initialize HandLandmarker for VIDEO mode (synchronous, for Streamlit)
        base_options = python.BaseOptions(model_asset_path=self.hand_landmarker_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp = 0
        
        print("RealTimeDetector initialized.")
    
    def _draw_landmarks(self, frame, hand_landmarks, frame_width, frame_height):
        """Draw hand landmarks on the frame."""
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_point = (int(start.x * frame_width), int(start.y * frame_height))
            end_point = (int(end.x * frame_width), int(end.y * frame_height))
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
    
    def predict(self, frame):
        """Process a single frame and return prediction. Used by Streamlit app."""
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB and create MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process frame synchronously (VIDEO mode)
        self.frame_timestamp += 33  # ~30fps
        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp)
        
        prediction = "Waiting..."
        confidence = 0.0
        
        # Data containers
        left_hand_data = np.zeros(63)
        right_hand_data = np.zeros(63)
        
        # Process detection results
        if result.hand_landmarks:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                # Draw landmarks
                self._draw_landmarks(frame, hand_landmarks, frame_width, frame_height)
                
                # Get handedness
                if idx < len(result.handedness):
                    handedness = result.handedness[idx]
                    hand_label = handedness[0].category_name.lower()
                else:
                    hand_label = 'right' if idx == 0 else 'left'
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                if hand_label == 'left':
                    left_hand_data = np.array(landmarks)
                else:
                    right_hand_data = np.array(landmarks)
            
            # Combine data for prediction
            input_data = np.concatenate([left_hand_data, right_hand_data])
            
            # Predict
            if self.model:
                try:
                    prediction = self.model.predict([input_data])[0]
                    proba = self.model.predict_proba([input_data])
                    confidence = np.max(proba)
                except Exception as e:
                    print(f"Prediction error: {e}")
        
        # Display prediction on frame
        cv2.rectangle(frame, (0, 0), (frame_width, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Pred: {prediction} ({confidence:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        return frame, prediction, confidence
    
    def run(self):
        """Run the real-time detector with webcam. Standalone mode."""
        cap = cv2.VideoCapture(0)
        
        print("Starting Real-time Detector (2-Hand)...")
        print("Press 'q' to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame, pred, conf = self.predict(frame)
            
            cv2.imshow('ISL Real-time Translator', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def close(self):
        """Close the landmarker."""
        if self.landmarker:
            self.landmarker.close()

def main():
    detector = RealTimeDetector()
    detector.run()
    detector.close()

if __name__ == "__main__":
    main()
