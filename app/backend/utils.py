import numpy as np
import shutil
import os
import json
from fastapi import UploadFile
from camera import Camera

def default_converter(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def save_temp_video(file: UploadFile) -> str:
    temp_video_path = f"cache/temp_{file.filename}"
    os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_video_path

def load_from_cache(cache_path: str):
    if os.path.exists(cache_path):
        print(f"Cache hit. Loading data from {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read cache file {cache_path}. Error: {e}")
            return None
    print("Cache miss.")
    return None

def save_to_cache(cache_path: str, data: dict):
    print(f"Saving computed data to cache at {cache_path}")
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(data, f, default=default_converter)
        print("Cache save successful.")
    except IOError as e:
        print(f"Error: Could not write to cache file {cache_path}. Error: {e}")

def moving_average(data, window_size=5):
    if not data:
        return []
    smoothed_data = []
    for i in range(len(data)):
        valid_points = [p for p in data[max(0, i - window_size + 1) : i + 1] if p is not None]
        if not valid_points:
            smoothed_data.append(None)
            continue
        avg_x = np.mean([p[0] for p in valid_points])
        avg_y = np.mean([p[1] for p in valid_points])
        avg_z = np.mean([p[2] for p in valid_points])
        smoothed_data.append((avg_x, avg_y, avg_z))
    return smoothed_data

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def link_actions_to_players(
    action_detections, 
    player_tracks, 
    ball_3d_positions, 
    camera: Camera, 
    calibration_params: dict,
    iou_threshold=0.2
):
    volleyball_events = []
    player_tracks_by_frame = {int(frame_id): detections for frame_id, detections in player_tracks.items()}

    for action in action_detections:
        action_frame = action.get("start_frame")
        action_box = action.get("box")

        if action_frame is None or action_box is None:
            volleyball_events.append(action)
            continue
        
        players_in_frame = player_tracks_by_frame.get(action_frame, [])
        best_match_player = None
        max_iou = 0

        for player in players_in_frame:
            player_box = player.get("box")
            if player_box:
                iou = calculate_iou(action_box, player_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match_player = player

        enriched_event = action.copy()
        if max_iou > iou_threshold and best_match_player is not None:
            enriched_event["player_id"] = best_match_player.get("player_id")

            action_name = action.get("action")
            ball_pos = None
            if action_frame < len(ball_3d_positions) and ball_3d_positions[action_frame] is not None:
                ball_pos = ball_3d_positions[action_frame]
            elif action_frame > 0 and (action_frame - 1) < len(ball_3d_positions) and ball_3d_positions[action_frame - 1] is not None:
                ball_pos = ball_3d_positions[action_frame - 1]
            elif (action_frame + 1) < len(ball_3d_positions) and ball_3d_positions[action_frame + 1] is not None:
                ball_pos = ball_3d_positions[action_frame + 1]

            if ball_pos:
                if action_name in ["spike", "set"]:
                    enriched_event["ball_height_m"] = round(ball_pos[2], 2)
                    if action_name == "set":
                        set_position = (-2/9) * ball_pos[0] + 4
                        enriched_event["set_position"] = round(set_position, 1)
            if action_name == "block":
                print("camera.rvec:", camera.rvec)
            if action_name == "block" and camera.rvec is not None:
                player_box = best_match_player.get("box")
                top_of_block_pixel = ((player_box[0] + player_box[2]) / 2, player_box[1])
                NET_HEIGHT_M = 2.43 
                block_3d_pos = camera.get_point_3d_position(
                    top_of_block_pixel,
                    reference_real_height_m=NET_HEIGHT_M,
                    z_scale_calibration=calibration_params['z_scale'],
                    x_sensitivity=calibration_params['x_sens'],
                    ground_plane_offset=calibration_params['g_offset']
                )
                if block_3d_pos:
                    enriched_event["block_height_m"] = round(block_3d_pos[2], 2)
        volleyball_events.append(enriched_event)

    return volleyball_events