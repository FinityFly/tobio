# action_classifier.py

import warnings
import os
import boto3
from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
import time
from collections import deque, Counter

class ActionClassifier:
    def __init__(self, s3_bucket, s3_key, local_model_path='models/best_actionclassifier.pt'):
        warnings.filterwarnings("ignore")
        load_dotenv()
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.local_model_path = local_model_path
        self.model = self._load_model()
        
        self.known_classes = {
            0: 'block', 
            1: 'defense', 
            2: 'serve', 
            3: 'set', 
            4: 'spike'
        }

    def _load_model(self):
        if not os.path.exists(self.local_model_path):
            os.makedirs(os.path.dirname(self.local_model_path), exist_ok=True)
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            s3 = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            s3.download_file(self.s3_bucket, self.s3_key, self.local_model_path)
        return YOLO(self.local_model_path)

    def classify_action(self, mp4_path, conf_thresh=0.5, sliding_window_size=3, action_cooldowns=None, default_cooldown=15, trigger_count=2):
        if action_cooldowns is None:
            action_cooldowns = {}
        print(f"Action cooldowns: {action_cooldowns}, Default cooldown: {default_cooldown}")

        action_detections = []
        video_metadata = {
            "fps": 0, "width": 0, "height": 0, "total_frames": 0,
        }
        
        try:
            cap = cv2.VideoCapture(mp4_path)
            video_metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
            video_metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_metadata["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Determine the maximum possible cooldown to size the buffer appropriately
            all_cooldowns = list(action_cooldowns.values())
            all_cooldowns.append(default_cooldown)
            max_cooldown = max(all_cooldowns) if all_cooldowns else default_cooldown

            print(f"Processing... FPS: {video_metadata['fps']} | Conf: {conf_thresh} | Win: {sliding_window_size} | Trig: {trigger_count} | Max Cooldown: {max_cooldown}")
            
            sliding_window = deque(maxlen=sliding_window_size)
            # The buffer's max length is the longest possible cooldown
            event_display_buffer = deque(maxlen=max_cooldown) 
            
            active_event_type = None
            active_event_start = 0
            active_event_box = None
            frame_id = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(frame, conf=conf_thresh, max_det=300, verbose=False)
                
                detected_classes_names = []
                current_frame_boxes = {} 

                if len(results[0]) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in self.known_classes:
                            name = self.known_classes[cls_id]
                        else:
                            name = f"unknown_{cls_id}"
                        
                        detected_classes_names.append(name)
                        current_frame_boxes[name] = box.xyxy[0].tolist() 
                    
                    sliding_window.extend(detected_classes_names)
                    
                    most_common = Counter(sliding_window).most_common(1)
                    if most_common:
                        cls_name, count = most_common[0]
                        
                        if count >= trigger_count and cls_name is not None:
                            box_to_store = current_frame_boxes.get(cls_name, None)

                            # --- MODIFIED LOGIC: PER-EVENT COOLDOWN ---
                            # 1. Get the specific cooldown for this action, or use the default.
                            cooldown_frames = action_cooldowns.get(cls_name, default_cooldown)
                            
                            # 2. Clear any previous event from the buffer.
                            event_display_buffer.clear()
                            
                            # 3. Refill the buffer with the new event for its specific cooldown period.
                            for _ in range(cooldown_frames):
                                event_display_buffer.append((cls_name, box_to_store))
                            # --- END OF MODIFICATION ---
                else:
                    sliding_window.append(None)

                current_frame_action = None
                current_frame_box = None
                
                if event_display_buffer:
                    current_frame_action, current_frame_box = event_display_buffer.popleft()

                if current_frame_action != active_event_type:
                    if active_event_type is not None:
                        action_detections.append({
                            "action": active_event_type,
                            "start_frame": active_event_start,
                            "end_frame": frame_id - 1,
                            "box": active_event_box
                        })
                    
                    if current_frame_action is not None:
                        active_event_start = frame_id
                        active_event_box = current_frame_box 
                    active_event_type = current_frame_action
                
                if active_event_type is not None and current_frame_box is not None:
                    active_event_box = current_frame_box

                frame_id += 1

            if active_event_type is not None:
                action_detections.append({
                    "action": active_event_type,
                    "start_frame": active_event_start,
                    "end_frame": frame_id - 1,
                    "box": active_event_box
                })

            cap.release()
            return {"action_detections": action_detections, "video_metadata": video_metadata}

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            return {"action_detections": [], "video_metadata": video_metadata}