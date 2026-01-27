import warnings
import os
import boto3
from ultralytics import YOLO
from dotenv import load_dotenv
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from court_estimator import CourtEstimator

class CourtTracker:
    def __init__(self, s3_bucket, s3_key, local_model_path='models/best_courttracker.pt'):
        warnings.filterwarnings("ignore")
        load_dotenv()
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.local_model_path = local_model_path
        self.model = self._load_model()
        print(f"Model Task: {self.model.model.task}")

        self.estimator = CourtEstimator()

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

    def track_court(self, mp4_path, court_class_idx=1, conf_thresh=0.3):
        avg_corners = None
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
            
            self.estimator.court_class_id = court_class_idx
            
            # =========================================================================
            # NEW LOGIC: Sample specific frames + Filter skewed courts + Fix Ordering
            # =========================================================================
            frames_to_sample = [0, 29, 59, 89, 119] # 1st, 30th, 60th, 90th, 120th frames
            sampled_corners = []
            
            # Tolerance for how "horizontal" the top/bottom lines must be. 
            # 10% of video height filters out the crazy diagonals seen in your image.
            y_tolerance = video_metadata['height'] * 0.05

            print(f"Sampling court from frames: {', '.join(str(f+1) for f in frames_to_sample)} (filtering skewed detections)...")

            for frame_id in frames_to_sample:
                if frame_id >= video_metadata["total_frames"]:
                    print(f"Warning: Frame {frame_id} is out of bounds. Total frames: {video_metadata['total_frames']}.")
                    continue
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {frame_id}.")
                    continue
                
                results = self.model(frame, verbose=False)[0]
                estimation = self.estimator.predict(results)
                
                if estimation['status'] == 'success' and estimation['confidence'] >= conf_thresh:
                    corners = estimation['corners'] # List of [x, y]
                    
                    # 1. Sort by Y to separate the Far side (Top) from Near side (Bottom)
                    # In pixel coords, Top is smaller Y, Bottom is larger Y.
                    corners_sorted_y = sorted(corners, key=lambda p: p[1])
                    top_pair = corners_sorted_y[:2]
                    bottom_pair = corners_sorted_y[2:]
                    
                    # 2. Check "Horizontality" (Y-difference)
                    # If the Y values of the top pair or bottom pair differ too much, it's a skewed/bad detection.
                    top_y_diff = abs(top_pair[0][1] - top_pair[1][1])
                    bottom_y_diff = abs(bottom_pair[0][1] - bottom_pair[1][1])
                    
                    if top_y_diff <= y_tolerance and bottom_y_diff <= y_tolerance:
                        # 3. Sort by X to ensure consistent Corner Ordering (TL, TR, BR, BL)
                        # This prevents "twisting" when averaging across frames.
                        top_pair_x = sorted(top_pair, key=lambda p: p[0])     # [TL, TR]
                        bottom_pair_x = sorted(bottom_pair, key=lambda p: p[0]) # [BL, BR]
                        
                        # Standardize order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
                        ordered_corners = [
                            top_pair_x[0], # TL
                            top_pair_x[1], # TR
                            bottom_pair_x[1], # BR
                            bottom_pair_x[0]  # BL
                        ]
                        sampled_corners.append(ordered_corners)
                
                frame_id += 1
            
            if sampled_corners:
                corners_array = np.array(sampled_corners)
                avg_corners = corners_array.mean(axis=0).tolist()
                print(f"Court estimated from {len(sampled_corners)} valid horizontal samples.")
            else:
                print("Could not find any valid horizontal court in the first 60 frames.")


            # DEBUGGING
            # fig = plt.figure(figsize=(8, 6))
            # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # plt.imshow(img_rgb)
            # if avg_corners is not None:
            #     avg_corners_int = np.array(avg_corners, dtype=np.int32)
            #     cv2.polylines(img_rgb, [avg_corners_int], True, (255, 0, 255), 3)
            #     for pt in avg_corners_int:
            #         x, y = pt
            #         cv2.circle(img_rgb, (x, y), 10, (255, 255, 255), -1)
            #         cv2.circle(img_rgb, (x, y), 6, (0, 0, 255), -1)
            # plt.imshow(img_rgb)
            # plt.title("Final Averaged Court Corners", fontsize=14)
            # plt.axis('off')
            # plt.show()



            

            # =========================================================================
            # OLD LOGIC (Commented Out): Process the whole video
            # =========================================================================
            """
            all_corners = []
            frame_id = 0
            start_time = time.time()
            last_second = start_time

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model(frame, verbose=False)[0]
                estimation = self.estimator.predict(results)
                
                if estimation['status'] == 'success' and estimation['confidence'] >= conf_thresh:
                    all_corners.append(estimation['corners'])

                frame_id += 1
                if frame_id % round(video_metadata['fps']) == 0:
                    batch_time = time.time() - last_second
                    avg_time_per_frame = batch_time / video_metadata['fps'] if video_metadata['fps'] else 0
                    remaining_frames = video_metadata['total_frames'] - frame_id
                    est_remaining_time = remaining_frames * avg_time_per_frame
                    print(f"Court Tracker: Frames {frame_id-round(video_metadata['fps'])}-{frame_id}/{video_metadata['total_frames']}: {batch_time:.3f}s (avg: {avg_time_per_frame:.3f}s/frame) - Estimated remaining time: {est_remaining_time:.3f}s")
                    last_second = time.time()
            
            total_time = time.time() - start_time
            print("---- Court Tracking Complete ----")
            print(f"Total processing time: {total_time:.3f}s for {frame_id} frames")

            if all_corners:
                # all_corners: list of [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ] per frame
                corners_array = np.array(all_corners)  # shape: (num_frames, 4, 2)
                avg_corners = corners_array.mean(axis=0).tolist()  # shape: (4, 2)
            """

            cap.release()

        except Exception as e:
            print(f"Error during court tracking: {e}")

        return {"court_corners": avg_corners, "video_metadata": video_metadata}