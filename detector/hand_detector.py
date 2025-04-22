import os
import cv2
import dlib
import time
import shutil
import yaml
import numpy as np


class HandDetector:
    def __init__(self, data_dir="dataset", model_path="models/hand_detector.svm", config_path="config/thresholds.yaml"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_file = os.path.join(data_dir, "annotations.txt")
        self.model_path = model_path
        self.config_path = config_path

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.detector = dlib.simple_object_detector(model_path) if os.path.exists(model_path) else None

        # Load gesture thresholds from YAML config
        self.left_threshold = 154
        self.right_threshold = 500
        self.up_threshold = 108
        self.down_threshold = 358

        self.load_thresholds()

    def load_thresholds(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.left_threshold = data.get("left_threshold", self.left_threshold)
                    self.right_threshold = data.get("right_threshold", self.right_threshold)
                    self.up_threshold = data.get("up_threshold", self.up_threshold)
                    self.down_threshold = data.get("down_threshold", self.down_threshold)
                print("Thresholds loaded into HandDetector.")
            except Exception as e:
                print(f"Failed to load thresholds: {e}")

    def collect_data(self, num_samples=200, clean_dataset=True):
        """Collect training data using a sliding window that moves only when a hand is detected in the center"""
        if clean_dataset and os.path.exists(self.images_dir):
            shutil.rmtree(self.images_dir)
            os.makedirs(self.images_dir)
            # Clear annotation file
            open(self.annotations_file, 'w').close()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Data Collection', cv2.WINDOW_NORMAL)
        
        # Sliding window parameters
        window_width, window_height = 190, 190
        x, y = 50, 50  # Start at a more convenient position
        x_step, y_step = 30, 40  # Reduced step size for slower movement
        frame_skip = 5  # Increased frame skip for slower capture
        current_frame = 0
        counter = 0
        
        # Define the center area threshold - hand must be within this percent of center
        center_threshold = 0.3  # 30% from center
        
        def is_hand_in_center(frame, x, y, w, h):
            """Check if hand is detected near the center of the window"""
            # Extract the region of interest
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                return False
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create a binary mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False
            
            # Find the largest contour (likely the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate center of hand
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return False
            
            # Calculate center of hand
            hand_cx = int(M["m10"] / M["m00"])
            hand_cy = int(M["m01"] / M["m00"])
            
            # Calculate center of window
            win_center_x = w // 2
            win_center_y = h // 2
            
            # Calculate distance from center (normalized by window dimensions)
            dist_x = abs(hand_cx - win_center_x) / win_center_x
            dist_y = abs(hand_cy - win_center_y) / win_center_y
            
            # Check if hand is within threshold distance from center
            return (dist_x < center_threshold and dist_y < center_threshold and 
                    cv2.contourArea(largest_contour) > 1000)  # Minimum area to filter noise
        
        # Function to detect hand in window (simpler version)
        def detect_hand_in_window(frame, x, y, w, h):
            # Extract the region of interest
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:  # Check if ROI is valid
                return False
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create a binary mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate the percentage of skin pixels
            skin_percent = (cv2.countNonZero(mask) / (w * h)) * 100
            
            # Return True if the percentage of skin pixels is above threshold
            return skin_percent > 15  # Adjust this threshold as needed
        
        # Open annotation file for writing
        with open(self.annotations_file, 'a') as f:
            print("Starting data collection. Press 'q' to quit.")
            print("Position your hand in the CENTER of the green rectangle...")
            
            # Wait a few seconds before starting
            time.sleep(5)  # Increased wait time
            
            wait_frames = 0
            center_stable_frames = 0  # Count frames with hand in center
            required_center_frames = 10  # Hand must be in center for this many frames
            
            while counter < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Create a copy of the original frame
                original = frame.copy()
                
                # Check if hand is in the current window
                hand_in_window = detect_hand_in_window(frame, x, y, window_width, window_height)
                hand_in_center = is_hand_in_center(frame, x, y, window_width, window_height)
                
                # Draw the sliding window 
                # Green if hand in center, yellow if hand detected but not centered, red if no hand
                if hand_in_center:
                    color = (0, 255, 0)  # Green for centered hand
                    cv2.putText(frame, "CENTERED!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                elif hand_in_window:
                    color = (0, 255, 255)  # Yellow for hand detected but not centered
                else:
                    color = (0, 0, 255)  # Red for no hand
                    
                cv2.rectangle(frame, (x, y), (x + window_width, y + window_height), color, 3)
                
                # Draw center target region
                center_x = x + window_width // 2
                center_y = y + window_height // 2
                target_width = int(window_width * center_threshold)
                target_height = int(window_height * center_threshold)
                cv2.rectangle(frame, 
                            (center_x - target_width//2, center_y - target_height//2),
                            (center_x + target_width//2, center_y + target_height//2),
                            (255, 0, 255), 2)  # Purple center target
                
                # Save image only when hand is detected in center for enough frames
                if hand_in_center:
                    center_stable_frames += 1
                    if center_stable_frames >= required_center_frames:
                        wait_frames += 1
                        # Add delay before capturing
                        if wait_frames >= 20:  # Increased wait time (20 frames)
                            current_frame += 1
                            if current_frame >= frame_skip:
                                current_frame = 0
                                
                                # Save the image
                                img_path = os.path.join(self.images_dir, f"{counter}.png")
                                cv2.imwrite(img_path, original)
                                
                                # Save annotation
                                f.write(f"{counter}:({x},{y},{x+window_width},{y+window_height}),")
                                f.flush()
                                
                                counter += 1
                                print(f"Saved sample {counter}/{num_samples}")
                                
                                # Move the window after successfully capturing an image
                                # Wait for additional frames before moving
                                time.sleep(0.5)  # Add pause between captures
                                
                                if x + window_width + x_step < frame.shape[1]:
                                    x += x_step
                                elif y + window_height + y_step < frame.shape[0]:
                                    y += y_step
                                    x = 50  # Reset x to starting position
                                else:
                                    x, y = 50, 50  # Reset to starting position
                                
                                # Reset counters after moving window
                                wait_frames = 0
                                center_stable_frames = 0
                else:
                    # Reset counters if hand moves out of center
                    center_stable_frames = 0
                    wait_frames = 0
                
                # Display instructions and progress
                cv2.putText(frame, f"Samples: {counter}/{num_samples}", (20, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if center_stable_frames > 0:
                    cv2.putText(frame, f"Center stable: {center_stable_frames}/{required_center_frames}", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                if hand_in_center and center_stable_frames >= required_center_frames:
                    cv2.putText(frame, f"Capturing in: {20-wait_frames}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif hand_in_window:
                    cv2.putText(frame, "Move hand to CENTER of box", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Place your hand in the box", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow('Data Collection', frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Data collection complete. {counter} samples saved.")      
 

    def preprocess_data(self, target_size=(300, 300), apply_augmentation=True):
        """Preprocess collected data with resizing, augmentation, and normalization"""
        if not os.path.exists(self.images_dir) or not os.path.exists(self.annotations_file):
            print("Error: Dataset not found. Run collect_data() first.")
            return None, None

        with open(self.annotations_file, "r") as f:
            annotation_text = f.read()

        annotation_dict = {}
        entries = annotation_text.split('),')
        for entry in entries:
            if not entry:
                continue
            parts = entry.split(':(')
            if len(parts) != 2:
                continue
            index = int(parts[0])
            coords = parts[1].replace(')', '')
            x1, y1, x2, y2 = map(int, coords.split(','))
            annotation_dict[index] = (x1, y1, x2, y2)

        images = []
        boxes = []

        for index, (x1, y1, x2, y2) in annotation_dict.items():
            img_path = os.path.join(self.images_dir, f"{index}.png")
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # Resize image and scale bbox
            resized_img = cv2.resize(img, target_size)
            scale_x = target_size[0] / w
            scale_y = target_size[1] / h
            new_x1 = int(x1 * scale_x)
            new_y1 = int(y1 * scale_y)
            new_x2 = int(x2 * scale_x)
            new_y2 = int(y2 * scale_y)

            dlib_box = [dlib.rectangle(left=new_x1, top=new_y1, right=new_x2, bottom=new_y2)]
            images.append(resized_img)
            boxes.append(dlib_box)

            if apply_augmentation:
                # Horizontal flip
                flipped_img = cv2.flip(resized_img, 1)
                flipped_x1 = target_size[0] - new_x2
                flipped_x2 = target_size[0] - new_x1
                dlib_flipped_box = [dlib.rectangle(left=flipped_x1, top=new_y1, right=flipped_x2, bottom=new_y2)]
                images.append(flipped_img)
                boxes.append(dlib_flipped_box)

                # Brightness adjustment
                bright_img = cv2.convertScaleAbs(resized_img, alpha=1.2, beta=30)
                images.append(bright_img)
                boxes.append(dlib_box)

                # Slight translation (e.g., shift down 5 pixels)
                M = np.float32([[1, 0, 0], [0, 1, 5]])
                translated_img = cv2.warpAffine(resized_img, M, target_size)
                translated_box = [dlib.rectangle(left=new_x1, top=new_y1+5, right=new_x2, bottom=new_y2+5)]
                images.append(translated_img)
                boxes.append(translated_box)

        print(f"Preprocessed {len(images)} images with augmentation.")
        return images, boxes

    def train_detector(self, test_split=0.2):
        """Train the hand detector using DLIB"""
        # Get preprocessed data
        images, boxes = self.preprocess_data()
        if not images or not boxes:
            print("Error: No data available for training.")
            return False

        # Calculate split point
        split_idx = int(len(images) * (1 - test_split))
        train_images = images[:split_idx]
        train_boxes = boxes[:split_idx]
        test_images = images[split_idx:]
        test_boxes = boxes[split_idx:]

        print(f"Training with {len(train_images)} images, testing with {len(test_images)} images.")

        # Configure training options
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = False
        options.C = 5
        options.num_threads = 4
        options.be_verbose = True

        # Train the detector
        print("Training detector... (this may take a while)")
        start_time = time.time()
        try:
            self.detector = dlib.train_simple_object_detector(train_images, train_boxes, options)
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds.")

            # Save the detector
            self.detector.save(self.model_path)
            print(f"Detector saved to {self.model_path}")

            # Evaluate the detector
            if test_images and test_boxes:
                print("Evaluating on training data:")
                train_results = dlib.test_simple_object_detector(train_images, train_boxes, self.detector)
                print("Training accuracy:", train_results)

                print("Evaluating on testing data:")
                test_results = dlib.test_simple_object_detector(test_images, test_boxes, self.detector)
                print("Testing accuracy:", test_results)

            return True
        except Exception as e:
            print(f"Error during training: {e}")
            return False

    def test_detector(self):
        """Test the detector with live webcam feed"""
        if not self.detector:
            self.load_thresholds()  # Load lại thresholds để đảm bảo mới nhất

            print("Error: No detector available. Train or load a detector first.")
            return
            
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Hand Detection Test', cv2.WINDOW_NORMAL)
        
        # Downscaling factor for faster detection
        scale_factor = 2.0
        
        # FPS calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        print("Testing detector. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
                
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Create a copy for processing
            copy = frame.copy()
            
            # Downsize for faster processing
            new_width = int(frame.shape[1]/scale_factor)
            new_height = int(frame.shape[0]/scale_factor)
            resized = cv2.resize(copy, (new_width, new_height))
            
            # Detect hand
            detections = self.detector(resized)
            
            # Variables for current detection
            center_x = 0
            center_y = 0
            gesture = "No Hand"
            
            # Process detections
            for detection in detections:
                # Scale coordinates back to original size
                x1 = int(detection.left() * scale_factor)
                y1 = int(detection.top() * scale_factor)
                x2 = int(detection.right() * scale_factor)
                y2 = int(detection.bottom() * scale_factor)
                
                # Calculate center
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Determine gesture
                if center_x < self.left_threshold:
                    gesture = "LEFT"
                elif center_x > self.right_threshold:
                    gesture = "RIGHT"
                elif center_y < self.up_threshold:
                    gesture = "UP"
                elif center_y > self.down_threshold:
                    gesture = "DOWN"
                else:
                    gesture = "NEUTRAL"
            
            # Draw threshold lines
            cv2.line(frame, (self.left_threshold, 0), (self.left_threshold, frame.shape[0]), (255, 0, 0), 2)
            cv2.line(frame, (self.right_threshold, 0), (self.right_threshold, frame.shape[0]), (0, 0, 255), 2)
            cv2.line(frame, (0, self.up_threshold), (frame.shape[1], self.up_threshold), (0, 255, 0), 2)
            cv2.line(frame, (0, self.down_threshold), (frame.shape[1], self.down_threshold), (0, 255, 255), 2)
            
            # Display information
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Center X: {center_x}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Center Y: {center_y}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Hand Detection Test', frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()