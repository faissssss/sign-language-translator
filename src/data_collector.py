import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import csv
import time

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

class DataCollector:
    def __init__(self, output_file='data/landmarks_data.csv', model_path='models/hand_landmarker.task'):
        self.output_file = output_file
        self.model_path = model_path
        self.latest_result = None
        self.result_timestamp = 0
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize CSV if it doesn't exist
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
    
    def _result_callback(self, result, output_image, timestamp_ms):
        """Callback to receive async detection results."""
        self.latest_result = result
        self.result_timestamp = timestamp_ms
    
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

    def collect(self):
        # Create HandLandmarker with LIVE_STREAM mode
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._result_callback
        )
        
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(0)
            current_label = "Waiting..."
            is_recording = False
            input_mode = False
            input_buffer = ""
            frames_recorded = 0
            frame_timestamp = 0
            
            print("Controls:")
            print("  Press 'ENTER' to change label")
            print("  Press 'SPACE' to start/stop recording")
            print("  Press 'q' to quit")

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Convert to RGB and create MediaPipe Image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    # Process frame asynchronously
                    frame_timestamp += 33  # ~30fps
                    landmarker.detect_async(mp_image, frame_timestamp)
                    
                    # UI Text
                    if input_mode:
                        status_text = f"TYPE NEW LABEL: {input_buffer}_"
                        sub_text = "Press ENTER to Finish"
                        color = (255, 255, 0)  # Cyan
                    else:
                        if is_recording:
                            status_text = f"RECORDING: {current_label}"
                            sub_text = f"Frames: {frames_recorded} | Press SPACE to Stop"
                            color = (0, 0, 255)  # Red
                        else:
                            status_text = f"Current Label: {current_label}"
                            sub_text = "Press SPACE to Record | Press ENTER to Change Label"
                            color = (0, 255, 0)  # Green
                    
                    # Draw main status
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    # Draw instructions below
                    cv2.putText(frame, sub_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # Data containers
                    left_hand_data = np.zeros(63)  # 21 * 3
                    right_hand_data = np.zeros(63)
                    hands_detected = False

                    # Process detection results
                    if self.latest_result is not None and self.latest_result.hand_landmarks:
                        hands_detected = True
                        for idx, hand_landmarks in enumerate(self.latest_result.hand_landmarks):
                            # Draw landmarks
                            self._draw_landmarks(frame, hand_landmarks, frame_width, frame_height)
                            
                            # Get handedness (Left or Right)
                            if idx < len(self.latest_result.handedness):
                                handedness = self.latest_result.handedness[idx]
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

                        if is_recording and not input_mode and current_label != "Waiting...":
                            # Combine data
                            full_row = [current_label]
                            full_row.extend(left_hand_data)
                            full_row.extend(right_hand_data)
                            
                            # Save to CSV
                            try:
                                with open(self.output_file, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(full_row)
                                frames_recorded += 1
                                print(f"Saved frame {frames_recorded} for label '{current_label}'")
                                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Recording indicator
                                cv2.putText(frame, "SAVED", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            except PermissionError:
                                print(f"ERROR: Could not write to {self.output_file}. Is it open in another program?")
                                cv2.putText(frame, "FILE ERROR!", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            except Exception as e:
                                print(f"ERROR: Failed to save data: {e}")

                    cv2.imshow('ISL Data Collector (2-Hand)', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if input_mode:
                        if key == 13:  # Enter
                            if input_buffer.strip():  # Only accept non-empty
                                current_label = input_buffer.strip()
                                input_mode = False
                                input_buffer = ""
                                frames_recorded = 0  # Reset counter for new label
                                print(f"Label set to: {current_label}")
                        elif key == 27:  # ESC to cancel
                            input_mode = False
                            input_buffer = ""
                        elif key == 8:  # Backspace
                            input_buffer = input_buffer[:-1]
                        elif key != 255:
                            # Only allow printable ASCII characters
                            if 32 <= key <= 126:
                                input_buffer += chr(key)
                    else:
                        if key == ord('q'):
                            break
                        elif key == 32:  # Space
                            if current_label == "Waiting...":
                                input_mode = True
                                print("PLEASE SET A LABEL FIRST! Press ENTER.")
                            else:
                                is_recording = not is_recording
                                if is_recording:
                                    frames_recorded = 0
                        elif key == 13:  # Enter
                            input_mode = True
                            is_recording = False
                            input_buffer = ""  # Start fresh

            except KeyboardInterrupt:
                print("\nUser interrupted (Ctrl+C). Exiting...")
            finally:
                cap.release()
                cv2.destroyAllWindows()
                print(f"Session ended. Data saved to {self.output_file}")

if __name__ == "__main__":
    # Backup old data if it exists and has different columns (simple check)
    if os.path.exists('data/landmarks_data.csv'):
        should_backup = False
        with open('data/landmarks_data.csv', 'r') as f:
            header = f.readline().strip().split(',')
            if len(header) < 100:  # Old format had ~64 columns
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
