import warnings
import os
from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
import time

class BallTracker:
    def __init__(self, s3_bucket, s3_key, local_model_path='models/best_balltracker.pt'):
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
                "Place the ball tracker weights there (e.g. best_balltracker.pt). S3 download is disabled."
            )
        return YOLO(self.local_model_path)

    def track_ball(self, mp4_path, ball_class_idx=0, conf_thresh=0.3):
        ball_tracks = []
        video_metadata = {
            "fps": 0,
            "width": 0,
            "height": 0,
            "total_frames": 0,
        }
        try:
            cap = cv2.VideoCapture(mp4_path)
            video_metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
            video_metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_metadata["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Video FPS: {video_metadata['fps']}, Total Frames: {video_metadata['total_frames']}")
            frame_id = 0
            start_time = time.time()
            last_second = start_time
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model(frame, verbose=False)
                for box in results[0].boxes:
                    if int(box.cls) == ball_class_idx and box.conf.item() >= conf_thresh:
                        ball_tracks.append({
                            'frame': frame_id,
                            'bbox': box.xyxy.tolist()[0],
                            'confidence': box.conf.item()
                        })
                        break 
                frame_id += 1
                if frame_id % round(video_metadata['fps']) == 0:
                    batch_time = time.time() - last_second
                    avg_time_per_frame = batch_time / video_metadata['fps'] if video_metadata['fps'] else 0
                    remaining_frames = video_metadata['total_frames'] - frame_id
                    est_remaining_time = remaining_frames * avg_time_per_frame
                    print(f"Ball Tracker: Frames {frame_id-round(video_metadata['fps'])}-{frame_id}/{video_metadata['total_frames']}: {batch_time:.3f}s (avg: {avg_time_per_frame:.3f}s/frame) - Estimated remaining time: {est_remaining_time:.3f}s")
                    last_second = time.time()
            cap.release()
            total_time = time.time() - start_time
            print("---- Ball Tracking Complete ----")
            print(f"Total processing time: {total_time:.3f}s for {frame_id} frames")

            print("---- Interpolating missing frames ----")
            start_interp_time = time.time()
            interpolated_tracks = self._interpolate_ball_tracks(ball_tracks, video_metadata['total_frames'])
            end_interp_time = time.time()
            print(f"Interpolation took {end_interp_time - start_interp_time:.3f}s")
            
            # Combine original and interpolated tracks, then sort by frame
            all_tracks = sorted(ball_tracks + interpolated_tracks, key=lambda x: x['frame'])

        except Exception as e:
            print(f"Error during ball tracking: {e}")
            
        return {"ball_tracks": all_tracks, "video_metadata": video_metadata}

    def _interpolate_ball_tracks(self, ball_tracks, total_frames, max_gap=10, max_dist=150):
        if not ball_tracks:
            return []

        detections = {track['frame']: track for track in ball_tracks}
        interpolated_tracks = []
        
        sorted_frames = sorted(detections.keys())
        
        for i in range(len(sorted_frames) - 1):
            frame_before = sorted_frames[i]
            frame_after = sorted_frames[i+1]
            
            gap = frame_after - frame_before
            if 1 < gap <= max_gap:
                bbox_before = detections[frame_before]['bbox']
                bbox_after = detections[frame_after]['bbox']
                
                center_before = ((bbox_before[0] + bbox_before[2]) / 2, (bbox_before[1] + bbox_before[3]) / 2)
                center_after = ((bbox_after[0] + bbox_after[2]) / 2, (bbox_after[1] + bbox_after[3]) / 2)
                
                dist = ((center_after[0] - center_before[0])**2 + (center_after[1] - center_before[1])**2)**0.5
                
                if dist <= max_dist:
                    for frame_num in range(frame_before + 1, frame_after):
                        # Linear interpolation
                        ratio = (frame_num - frame_before) / gap
                        
                        center_x = center_before[0] + ratio * (center_after[0] - center_before[0])
                        center_y = center_before[1] + ratio * (center_after[1] - center_before[1])
                        
                        w_before = bbox_before[2] - bbox_before[0]
                        h_before = bbox_before[3] - bbox_before[1]
                        w_after = bbox_after[2] - bbox_after[0]
                        h_after = bbox_after[3] - bbox_after[1]
                        
                        w = w_before + ratio * (w_after - w_before)
                        h = h_before + ratio * (h_after - h_before)
                        
                        x1 = center_x - w / 2
                        y1 = center_y - h / 2
                        x2 = center_x + w / 2
                        y2 = center_y + h / 2
                        
                        interpolated_tracks.append({
                            'frame': frame_num,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': 0.0, # mark as interpolated
                            'interpolated': True
                        })

        return interpolated_tracks