import warnings
import os
from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
import time
import numpy as np

class ServeRecognizer:
    def __init__(self, s3_bucket, s3_key, local_model_path='models/best_serverecognizer.pt'):
        warnings.filterwarnings("ignore")
        load_dotenv()
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.local_model_path = local_model_path
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.local_model_path):
            raise FileNotFoundError(
                f"Model not found at {self.local_model_path}. "
                "Place the serve recognizer weights there (e.g. best_serverecognizer.pt). S3 download is disabled."
            )
        return YOLO(self.local_model_path)

    def _order_points(self, pts):
        """
        point order: tl, tr, br, bl
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def recognize_serves(self, mp4_path, court_corners=None, serve_class_idx=0, conf_thresh=0.7, cooldown_frames=20):
        serve_events = []
        video_metadata = {}
        
        middle_court_y = None
        if court_corners and len(court_corners) == 4:
            try:
                # point order: tl, tr, br, bl
                ordered_corners = self._order_points(np.array(court_corners, dtype="float32"))
                top_y_avg = (ordered_corners[0][1] + ordered_corners[1][1]) / 2
                bottom_y_avg = (ordered_corners[2][1] + ordered_corners[3][1]) / 2
                middle_court_y = (top_y_avg + bottom_y_avg) / 2
                print(f"Calculated middle court line Y: {middle_court_y}")
            except Exception as e:
                print(f"Could not calculate middle court line: {e}")

        try:
            cap = cv2.VideoCapture(mp4_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_metadata = {
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height
            }

            print(f"Serve Recognizer started. FPS: {fps}, Total Frames: {total_frames}")
            frame_id = 0
            start_time = time.time()
            last_second = start_time

            # temporal classifier
            in_serve_sequence = False
            current_serve_start = 0
            current_serving_team = "Unknown"
            frames_since_last_positive = 0
            
            def end_serve_event(end_frame, serving_team):
                timestamp = current_serve_start / fps if fps > 0 else 0
                serve_events.append({
                    "start_frame": current_serve_start,
                    "end_frame": end_frame,
                    "timestamp": float(f"{timestamp:.2f}"),
                    "label": "serve",
                    "serving_team": serving_team
                })

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model(frame, verbose=False)

                is_positive_frame = False
                serve_box = None
                for box in results[0].boxes:
                    if int(box.cls) == serve_class_idx and box.conf.item() >= conf_thresh:
                        is_positive_frame = True
                        serve_box = box.xyxy[0].cpu().numpy()
                        break
                
                if is_positive_frame:
                    if not in_serve_sequence:
                        in_serve_sequence = True
                        current_serve_start = frame_id
                        
                        # classify serving team based on position relative to midline
                        if middle_court_y is not None and serve_box is not None:
                            serve_bottom_y = serve_box[3]  # y2 of the bounding box
                            if serve_bottom_y < middle_court_y:
                                current_serving_team = 1  # FAR team
                            else:
                                current_serving_team = 0  # NEAR team
                        else:
                            current_serving_team = "Unknown"
                    
                    frames_since_last_positive = 0
                else:
                    if in_serve_sequence:
                        frames_since_last_positive += 1
                        if frames_since_last_positive > cooldown_frames:
                            in_serve_sequence = False
                            end_frame = frame_id - frames_since_last_positive
                            end_serve_event(end_frame, current_serving_team)
                            current_serving_team = -1  # Reset for next event

                frame_id += 1
                if frame_id % round(video_metadata.get('fps', 30)) == 0:
                    batch_time = time.time() - last_second
                    avg_time_per_frame = batch_time / video_metadata['fps'] if video_metadata.get('fps', 0) > 0 else 0
                    remaining_frames = video_metadata['total_frames'] - frame_id
                    est_remaining_time = remaining_frames * avg_time_per_frame
                    print(f"Serve Recognizer: Frames {frame_id-round(video_metadata.get('fps', 30))}-{frame_id}/{video_metadata['total_frames']}: {batch_time:.3f}s (avg: {avg_time_per_frame:.3f}s/frame) - Estimated remaining time: {est_remaining_time:.3f}s")
                    last_second = time.time()

            if in_serve_sequence:
                end_serve_event(frame_id - 1, current_serving_team)

            cap.release()
            total_time = time.time() - start_time
            print(f"---- Serve Recognition Complete: Found {len(serve_events)} serves in {total_time:.3f}s ----")

        except Exception as e:
            print(f"Error during serve recognition: {e}")

        return {
            "serve_events": serve_events,
            "video_metadata": video_metadata
        }