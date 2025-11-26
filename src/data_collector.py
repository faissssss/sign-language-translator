import cv2
import mediapipe as mp
import numpy as np
import os
import csv

class DataCollector:
    def __init__(self, output_file='data/landmarks_data.csv'):
        self.output_file = output_file
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, # Changed to 2 hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize CSV if it doesn't exist or if we are starting fresh
        if not os.path.exists(output_file):
            self._init_csv()
            print(f"Created new dataset file: {output_file}")
        else:
            print(f"Found existing dataset: {output_file}")
            print("New data will be APPENDED to this file.")

    def _init_csv(self):
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header: label, 
            # Left Hand: lx0, ly0, lz0 ... lx20, ly20, lz20
            # Right Hand: rx0, ry0, rz0 ... rx20, ry20, rz20
            header = ['label']
            for hand in ['left', 'right']:
                for i in range(21):
                    header.extend([f'{hand}_x{i}', f'{hand}_y{i}', f'{hand}_z{i}'])
            writer.writerow(header)

    def collect(self):
        cap = cv2.VideoCapture(0)
        current_label = "Waiting..."
        is_recording = False
        input_mode = False
        input_buffer = ""
        frames_recorded = 0
        
        print("Controls:")
        print("  Press 'ENTER' to change label")
        print("  Press 'SPACE' to start/stop recording")
        print("  Press 'q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # UI Text
            if input_mode:
                status_text = f"TYPE NEW LABEL: {input_buffer}_"
                sub_text = "Press ENTER to Finish"
                color = (255, 255, 0) # Cyan
            else:
                if is_recording:
                    status_text = f"RECORDING: {current_label}"
                    sub_text = f"Frames: {frames_recorded} | Press SPACE to Stop"
                    color = (0, 0, 255) # Red
                else:
                    status_text = f"Current Label: {current_label}"
                    sub_text = "Press SPACE to Record | Press ENTER to Change Label"
                    color = (0, 255, 0) # Green
            
            # Draw main status
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Draw instructions below
            cv2.putText(frame, sub_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Data containers
            left_hand_data = np.zeros(63) # 21 * 3
            right_hand_data = np.zeros(63)
            hands_detected = False

            if results.multi_hand_landmarks and results.multi_handedness:
                hands_detected = True
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get hand label (Left or Right)
                    hand_label = handedness.classification[0].label.lower()
                    
                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    if hand_label == 'left':
                        left_hand_data = np.array(landmarks)
                    else:
                        right_hand_data = np.array(landmarks)

                if is_recording and not input_mode and current_label != "Waiting...":
                    # Combine data
                    full_row = [current_label]
                    full_row.extend(left_hand_data)
                    full_row.extend(right_hand_data)
                    
                    # Save to CSV
                    with open(self.output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(full_row)
                    
                    frames_recorded += 1
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) # Recording indicator

            cv2.imshow('ISL Data Collector (2-Hand)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if input_mode:
                if key == 13: # Enter
                    if input_buffer.strip(): # Only accept non-empty
                        current_label = input_buffer.strip()
                        input_mode = False
                        input_buffer = ""
                        frames_recorded = 0 # Reset counter for new label
                        print(f"Label set to: {current_label}")
                elif key == 27: # ESC to cancel
                    input_mode = False
                    input_buffer = ""
                elif key == 8: # Backspace
                    input_buffer = input_buffer[:-1]
                elif key != 255:
                    # Only allow printable ASCII characters
                    if 32 <= key <= 126:
                        input_buffer += chr(key)
            else:
                if key == ord('q'):
                    break
                elif key == 32: # Space
                    if current_label != "Waiting...":
                        is_recording = not is_recording
                        if is_recording:
                            frames_recorded = 0
                elif key == 13: # Enter
                    input_mode = True
                    is_recording = False
                    input_buffer = "" # Start fresh

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Backup old data if it exists and has different columns (simple check)
    if os.path.exists('data/landmarks_data.csv'):
        should_backup = False
        with open('data/landmarks_data.csv', 'r') as f:
            header = f.readline().strip().split(',')
            if len(header) < 100: # Old format had ~64 columns
                should_backup = True
        
        if should_backup:
            print("Backing up old dataset to data/landmarks_data_old.csv")
            import shutil
            try:
                shutil.move('data/landmarks_data.csv', 'data/landmarks_data_old.csv')
            except Exception as e:
                print(f"Warning: Could not backup file: {e}")
                
    collector = DataCollector()
    collector.collect()
