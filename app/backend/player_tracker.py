import warnings
import os
from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from scipy.spatial.distance import cdist

class PlayerTracker:
    def __init__(self, s3_bucket, s3_key, local_model_path='models/best_playertracker.pt'):
        warnings.filterwarnings("ignore")
        load_dotenv()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.local_model_path = local_model_path
        self.model = self._load_yolo_model()
        self.target_class_id = 0 

        self.reid_model = self._load_reid_model()
        self.reid_preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)), 
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.player_gallery = {}
        self.id_mapping = {}
        self.next_unique_id = 1
        self.prev_frame_gray = None
        
        self.initial_assignment_done = False
        self.best_initialization_frame_data = {'frame': None, 'detections': []} # Initialize with empty data
        self.INITIALIZATION_WINDOW_FRAMES = 150

    def _load_yolo_model(self):
        if not os.path.exists(self.local_model_path):
            raise FileNotFoundError(
                f"Model not found at {self.local_model_path}. "
                "Place the player tracker weights there (e.g. best_playertracker.pt). S3 download is disabled."
            )
        return YOLO(self.local_model_path)

    def _load_reid_model(self):
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        base_model.fc = nn.Identity()
        base_model.to(self.device)
        base_model.eval()
        return base_model

    def _detect_scene_change(self, frame):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.resize(current_gray, (128, 72))
        is_cut = False
        if self.prev_frame_gray is not None:
            diff = cv2.absdiff(current_gray, self.prev_frame_gray)
            non_zero_count = np.count_nonzero(diff > 30)
            if non_zero_count > (current_gray.size * 0.40):
                is_cut = True
        self.prev_frame_gray = current_gray
        return is_cut

    def _is_near_side_player(self, box, frame_height):
        x1, y1, x2, y2 = box
        box_h = y2 - y1
        box_center_y = (y1 + y2) / 2
        if box_center_y < (frame_height * 0.35) or box_h < (frame_height * 0.08):
            return False
        return True

    def _get_embedding(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = frame.shape
        pad = 5
        x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1: return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        img_tensor = self.reid_preprocess(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.reid_model(img_tensor).cpu().numpy()

    def _calculate_centrality_penalty(self, box, frame_width, max_penalty=0.15):
        box_center_x = (box[0] + box[2]) / 2
        deviation = abs(box_center_x - frame_width / 2) / (frame_width / 2)
        return max_penalty * (deviation ** 2)

    def _resolve_identity(self, yolo_id, embedding, assigned_ids, threshold, max_players, box, frame_width, ambiguity_threshold=0.85):
        if yolo_id in self.id_mapping and self.id_mapping[yolo_id] not in assigned_ids:
            return self.id_mapping[yolo_id]

        centrality_penalty = self._calculate_centrality_penalty(box, frame_width)
        matches = []
        for pid, data in self.player_gallery.items():
            if pid in assigned_ids: continue
            gallery_mean = np.mean(np.array(data['embeddings'][-20:]), axis=0).reshape(1, -1)
            dist = cdist(embedding, gallery_mean, metric='cosine')[0][0]
            matches.append((dist + centrality_penalty, pid))

        if not matches:
            if len(self.player_gallery) < max_players and centrality_penalty < 0.1:
                new_id = self.next_unique_id
                self.next_unique_id += 1
                self.id_mapping[yolo_id] = new_id
                self.player_gallery[new_id] = {'embeddings': [embedding], 'last_seen': 0}
                return new_id
            return None

        matches.sort(key=lambda x: x[0])
        best_dist, best_pid = matches[0]
        
        is_ambiguous = len(matches) > 1 and (best_dist / matches[1][0] > ambiguity_threshold)

        if best_dist < threshold and not is_ambiguous:
            self.id_mapping[yolo_id] = best_pid
            return best_pid
        elif not is_ambiguous and len(self.player_gallery) < max_players and centrality_penalty < 0.1:
            new_id = self.next_unique_id
            self.next_unique_id += 1
            self.id_mapping[yolo_id] = new_id
            self.player_gallery[new_id] = {'embeddings': [embedding], 'last_seen': 0}
            return new_id
        
        return None
    
    def _perform_initial_assignment(self, initialization_data, max_players, frame_id):
        if not initialization_data or not initialization_data['detections']:
            print("No suitable frame found for initial assignment. Skipping.")
            self.initial_assignment_done = True # dont rerun
            return

        frame, detections = initialization_data['frame'], initialization_data['detections']
        print(f"Performing initial player ID assignment using data from best frame (found {len(detections)} players).")

        detections.sort(key=lambda d: d['box'][3])
        assigned_id = 1
        for det in detections:
            if len(self.player_gallery) >= max_players: break
            embedding = self._get_embedding(frame, det['box'])
            if embedding is not None:
                self.player_gallery[assigned_id] = {'embeddings': [embedding], 'last_seen': frame_id}
                if 'yolo_id' in det:
                    self.id_mapping[det['yolo_id']] = assigned_id
                assigned_id += 1
        
        if self.player_gallery:
            self.next_unique_id = assigned_id
            print(f"Successfully assigned {len(self.player_gallery)} initial player IDs.")
        self.initial_assignment_done = True

    def track_players(self, mp4_path, conf_thresh=0.3, reid_sim_threshold=0.25, max_unique_players=12, stale_after_frames=150):
        player_tracks = {}
        cap = cv2.VideoCapture(mp4_path)
        video_metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        print(f"Tracking... Cap: {max_unique_players} Players | Sim Thresh: {reid_sim_threshold}")

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self._detect_scene_change(frame):
                print(f"Hard Cut detected at frame {frame_id}. Resetting spatial tracking.")
                self.id_mapping = {} 

            results = self.model.track(frame, persist=True, conf=conf_thresh, tracker="botsort.yaml", classes=[self.target_class_id], verbose=False)
            
            if not self.initial_assignment_done:
                if frame_id <= self.INITIALIZATION_WINDOW_FRAMES:
                    current_detections = []
                    if results[0].boxes.id is not None:
                         for box_obj, yolo_id in zip(results[0].boxes, results[0].boxes.id.int().cpu().tolist()):
                            box = box_obj.xyxy.cpu().tolist()[0]
                            if self._is_near_side_player(box, video_metadata["height"]):
                                current_detections.append({'box': box, 'yolo_id': yolo_id})
                    if len(current_detections) > len(self.best_initialization_frame_data['detections']):
                        self.best_initialization_frame_data = {
                            'frame': frame.copy(),
                            'detections': current_detections
                        }
                        print(f"Frame {frame_id}: Found new best frame for initialization with {len(current_detections)} players.")
                else:
                    self._perform_initial_assignment(self.best_initialization_frame_data, max_unique_players, frame_id)

            # MAIN
            frame_detections = []
            if self.initial_assignment_done and results[0].boxes.id is not None:
                assigned_ids_this_frame = set()
                for box_obj, raw_yolo_id in zip(results[0].boxes, results[0].boxes.id.int().cpu().tolist()):
                    if not self._is_near_side_player(box_obj.xyxy.cpu().tolist()[0], video_metadata["height"]):
                        continue

                    embedding = self._get_embedding(frame, box_obj.xyxy.cpu().tolist()[0])
                    if embedding is None: continue
                    
                    persistent_id = self._resolve_identity(raw_yolo_id, embedding, assigned_ids_this_frame, reid_sim_threshold, max_unique_players, box_obj.xyxy.cpu().tolist()[0], video_metadata["width"])
                    if persistent_id is not None:
                        assigned_ids_this_frame.add(persistent_id)
                        frame_detections.append({"player_id": persistent_id, "box": box_obj.xyxy.cpu().tolist()[0], "confidence": round(box_obj.conf.cpu().tolist()[0], 2)})
                        self.player_gallery[persistent_id]['embeddings'].append(embedding)
                        self.player_gallery[persistent_id]['last_seen'] = frame_id
                        if len(self.player_gallery[persistent_id]['embeddings']) > 50:
                            self.player_gallery[persistent_id]['embeddings'].pop(0)
            
            player_tracks[frame_id] = frame_detections
            
            if frame_id % 100 == 0 and self.initial_assignment_done:
                stale_ids = [pid for pid, data in self.player_gallery.items() if frame_id - data['last_seen'] > stale_after_frames]
                for pid in stale_ids:
                    del self.player_gallery[pid]
                    for y_id, p_id in list(self.id_mapping.items()):
                        if p_id == pid:
                            del self.id_mapping[y_id]
                if stale_ids:
                    print(f"Pruned {len(stale_ids)} stale player IDs.")

            frame_id += 1
            if frame_id % 30 == 0:
                 print(f"Player Tracker: Frame {frame_id}/{video_metadata['total_frames']} | Active IDs: {len(self.player_gallery)}")

        cap.release()
        return {"player_tracks": player_tracks, "video_metadata": video_metadata}