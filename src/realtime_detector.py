import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

class RealTimeDetector:
    def __init__(self, model_path='models/isl_classifier.p'):
        try:
            self.model_dict = pickle.load(open(model_path, 'rb'))
            self.model = self.model_dict['model']
        except FileNotFoundError:
            print("Model not found. Please train the model first.")
            self.model = None
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, # 2 hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def predict(self, frame):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        prediction = "Waiting..."
        confidence = 0.0
        
        # Data containers
        left_hand_data = np.zeros(63)
        right_hand_data = np.zeros(63)
        hands_detected = False
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hands_detected = True
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get label
                hand_label = handedness.classification[0].label.lower()
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                if hand_label == 'left':
                    left_hand_data = np.array(landmarks)
                else:
                    right_hand_data = np.array(landmarks)
            
            # Combine
            input_data = np.concatenate([left_hand_data, right_hand_data])
            
            # Predict
            if self.model:
                try:
                    prediction = self.model.predict([input_data])[0]
                    proba = self.model.predict_proba([input_data])
                    confidence = np.max(proba)
                    
                    # Display
                    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
                    cv2.putText(frame, f"Pred: {prediction} ({confidence:.2f})", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                except Exception as e:
                    print(f"Prediction error: {e}")
        
        return frame, prediction, confidence

def main():
    detector = RealTimeDetector()
    cap = cv2.VideoCapture(0)
    
    print("Starting Real-time Detector (2-Hand)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame, pred, conf = detector.predict(frame)
        
        cv2.imshow('ISL Real-time Translator', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
