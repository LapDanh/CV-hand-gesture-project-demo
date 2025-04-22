import os
import cv2
import dlib
import time
import pyautogui
import yaml


class GestureController:
    def __init__(self, detector, config_path="config/thresholds.yaml"):
        self.detector = detector
        self.config_path = config_path
        self.left_threshold = 154
        self.right_threshold = 500
        self.up_threshold = 108  # Giá trị y thấp (phía trên màn hình)
        self.down_threshold = 358  # Giá trị y cao (phía dưới màn hình)
        self.load_thresholds()
        self.last_key = None
        self.key_cooldown = 0
        self.cooldown_time = 10

    def load_thresholds(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.left_threshold = data.get("left_threshold", self.left_threshold)
                    self.right_threshold = data.get("right_threshold", self.right_threshold)
                    self.up_threshold = data.get("up_threshold", self.up_threshold)
                    self.down_threshold = data.get("down_threshold", self.down_threshold)
                print("Calibration loaded successfully.")
            except Exception as e:
                print(f"Error loading calibration: {e}")
        else:
            print("No calibration file found.")

    def save_thresholds(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump({
                "left_threshold": self.left_threshold,
                "right_threshold": self.right_threshold,
                "up_threshold": self.up_threshold,
                "down_threshold": self.down_threshold,
            }, f)
        print("Calibration saved.")
    def calibrate(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        scale_factor = 2.0
        print("Calibration mode active. Press l/r/u/d to calibrate, q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            resized = cv2.resize(frame, (frame.shape[1] // int(scale_factor), frame.shape[0] // int(scale_factor)))
            detections = self.detector(resized)
            center_x = 0
            center_y = 0  # Thêm biến theo dõi tọa độ Y

            for detection in detections:
                x1 = int(detection.left() * scale_factor)
                y1 = int(detection.top() * scale_factor)
                x2 = int(detection.right() * scale_factor)
                y2 = int(detection.bottom() * scale_factor)
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2  # Tính tọa độ Y của tâm
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw threshold lines for left/right
            cv2.line(frame, (self.left_threshold, 0), (self.left_threshold, frame.shape[0]), (255, 0, 0), 2)
            cv2.line(frame, (self.right_threshold, 0), (self.right_threshold, frame.shape[0]), (0, 0, 255), 2)
            
            # Draw threshold lines for up/down
            cv2.line(frame, (0, self.up_threshold), (frame.shape[1], self.up_threshold), (0, 255, 0), 2)
            cv2.line(frame, (0, self.down_threshold), (frame.shape[1], self.down_threshold), (0, 255, 255), 2)

            # Display info
            cv2.putText(frame, f"Center X: {center_x}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Center Y: {center_y}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l') and center_x > 0:
                self.left_threshold = center_x
                print(f"Left threshold set to {center_x}")
            elif key == ord('r') and center_x > 0:
                self.right_threshold = center_x
                print(f"Right threshold set to {center_x}")
            elif key == ord('u') and center_y > 0:
                self.up_threshold = center_y
                print(f"Up threshold set to {center_y}")
            elif key == ord('d') and center_y > 0:
                self.down_threshold = center_y
                print(f"Down threshold set to {center_y}")

        cap.release()
        cv2.destroyAllWindows()
        self.save_thresholds()
        
    def detect_and_control(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Game Controller', cv2.WINDOW_NORMAL)
        scale_factor = 2.0
        frame_count = 0
        start_time = time.time()
        fps = 0

        print("Game controller active. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()

            frame = cv2.flip(frame, 1)
            resized = cv2.resize(frame, (frame.shape[1] // int(scale_factor), frame.shape[0] // int(scale_factor)))
            detections = self.detector(resized)
            center_x = 0
            center_y = 0  # Thêm biến theo dõi tọa độ Y
            gesture = "No Hand"
            current_key = None

            for detection in detections:
                x1 = int(detection.left() * scale_factor)
                y1 = int(detection.top() * scale_factor)
                x2 = int(detection.right() * scale_factor)
                y2 = int(detection.bottom() * scale_factor)

                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2  # Tính tọa độ Y của tâm

                if center_x < self.left_threshold:
                    gesture = "LEFT"
                    current_key = "left"
                elif center_x > self.right_threshold:
                    gesture = "RIGHT"
                    current_key = "right"
                elif center_y < self.up_threshold:  # Thay đổi logic ở đây - nếu y nhỏ (ở phía trên)
                    gesture = "UP"
                    current_key = "up"
                elif center_y > self.down_threshold:  # Thay đổi logic ở đây - nếu y lớn (ở phía dưới)
                    gesture = "DOWN"
                    current_key = "down"
                else:
                    gesture = "NEUTRAL"
                    current_key = None

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if current_key and (current_key != self.last_key or self.key_cooldown <= 0):
                pyautogui.press(current_key)
                self.last_key = current_key
                self.key_cooldown = self.cooldown_time

            if self.key_cooldown > 0:
                self.key_cooldown -= 1

            # Draw threshold lines
            cv2.line(frame, (self.left_threshold, 0), (self.left_threshold, frame.shape[0]), (255, 0, 0), 2)
            cv2.line(frame, (self.right_threshold, 0), (self.right_threshold, frame.shape[0]), (0, 0, 255), 2)
            cv2.line(frame, (0, self.up_threshold), (frame.shape[1], self.up_threshold), (0, 255, 0), 2)
            cv2.line(frame, (0, self.down_threshold), (frame.shape[1], self.down_threshold), (0, 255, 255), 2)

            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Key: {current_key if current_key else 'None'}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Center Y: {center_y}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Game Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()