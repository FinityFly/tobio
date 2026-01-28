import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
import shutil
import os
import secrets
import json
import utils
from court_tracker import CourtTracker
from ball_tracker import BallTracker
from action_classifier import ActionClassifier
from serve_recognizer import ServeRecognizer
from player_tracker import PlayerTracker
from camera import Camera


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://tobio.daniellu.ca",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_locations = {
    "court_tracker": {
        "s3_bucket": "tobio-models",
        "s3_key": "yolov11-court-seg-v2/weights/best.pt"
    },
    "ball_tracker": {
        "s3_bucket": "tobio-models",
        "s3_key": "yolov11-volleyball-v2/weights/best.pt"
    },
    "action_classifier": {
        "s3_bucket": "tobio-models",
        "s3_key": "yolov11-actions-v2/weights/best.pt"
    },
    "serve_recognizer": {
        "s3_bucket": "tobio-models",
        "s3_key": "yolov11-serve/weights/best.pt" 
    },
    "player_tracker": {
        "s3_bucket": "tobio-models",
        "s3_key": "yolov11-player-fixed/weights/best.pt" 
    }
}

security = HTTPBasic()
USERNAME = os.getenv("API_USERNAME", "tobio")
PASSWORD = os.getenv("API_PASSWORD", "tobio")

# Lazy-loaded so the app starts quickly; models load on first /process-video/ use (local files only, no S3)
_action_classifier = None
_court_tracker = None
_ball_tracker = None
_serve_recognizer = None
_player_tracker = None


def _get_action_classifier():
    global _action_classifier
    if _action_classifier is None:
        try:
            _action_classifier = ActionClassifier(
                s3_locations["action_classifier"]["s3_bucket"],
                s3_locations["action_classifier"]["s3_key"],
                local_model_path="models/best_actionclassifier.pt",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Action model failed to load: {e}",
            )
    return _action_classifier


def _get_court_tracker():
    global _court_tracker
    if _court_tracker is None:
        try:
            _court_tracker = CourtTracker(
                s3_locations["court_tracker"]["s3_bucket"],
                s3_locations["court_tracker"]["s3_key"],
                local_model_path="models/best_courttracker.pt",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Court model failed to load: {e}",
            )
    return _court_tracker


def _get_ball_tracker():
    global _ball_tracker
    if _ball_tracker is None:
        try:
            _ball_tracker = BallTracker(
                s3_locations["ball_tracker"]["s3_bucket"],
                s3_locations["ball_tracker"]["s3_key"],
                local_model_path="models/best_balltracker.pt",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ball model failed to load: {e}",
            )
    return _ball_tracker


def _get_serve_recognizer():
    global _serve_recognizer
    if _serve_recognizer is None:
        try:
            _serve_recognizer = ServeRecognizer(
                s3_locations["serve_recognizer"]["s3_bucket"],
                s3_locations["serve_recognizer"]["s3_key"],
                local_model_path="models/best_serverecognizer.pt",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Serve model failed to load: {e}",
            )
    return _serve_recognizer


def _get_player_tracker():
    global _player_tracker
    if _player_tracker is None:
        try:
            _player_tracker = PlayerTracker(
                s3_locations["player_tracker"]["s3_bucket"],
                s3_locations["player_tracker"]["s3_key"],
                local_model_path="models/best_playertracker.pt",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Player model failed to load: {e}",
            )
    return _player_tracker

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, USERNAME)
    is_correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/routes")
def list_routes():
    """Debug: list registered routes (helps verify backend URL and that /process-court-lines/ exists)."""
    out = []
    for r in app.routes:
        if hasattr(r, "path"):
            methods = list(getattr(r, "methods", set()) or [])
            out.append({"path": r.path, "methods": methods})
    return {"routes": out}

@app.post("/process-court-lines/")
@app.post("/process-court-lines")
def process_court_lines_endpoint(file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    temp_video_path = utils.save_temp_video(file)
    # This endpoint still calls the court tracker to get the initial estimate
    tracking_data = CourtTracker(
        s3_bucket=s3_locations["court_tracker"]["s3_bucket"],
        s3_key=s3_locations["court_tracker"]["s3_key"]
    ).track_court(temp_video_path)
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    return tracking_data


@app.post("/process-video/")
@app.post("/process-video")
def process_video_endpoint(
    file: UploadFile = File(...),
    court_corners: str = Form(None),
    camera_height: float = Form(7.0),
    focal_length: float = Form(2.0),
    ball_height_calibration: float = Form(1.0),
    ball_side_calibration: float = Form(2.0),
    ground_plane_offset: float = Form(0.0),
    username: str = Depends(verify_credentials)
):
    temp_video_path = utils.save_temp_video(file)
    base_filename, _ = os.path.splitext(file.filename)
    cache_file_path = f"cache/{base_filename}.json"

    calibration_params = {
        'z_scale': ball_height_calibration,
        'x_sens': ball_side_calibration,
        'g_offset': ground_plane_offset
    }

    camera = Camera(camera_height_m=camera_height)
    user_court_corners = json.loads(court_corners) if court_corners else None
    
    # Calibrate camera if we have corners. This calibration will be used later.
    if user_court_corners:
        # We need video metadata for calibration, which might be in the cache
        temp_meta = {}
        cached_data = utils.load_from_cache(cache_file_path)
        if cached_data and "video_metadata" in cached_data:
            temp_meta = cached_data["video_metadata"]
        else:
            # Quick probe if not cached (less efficient but necessary)
            cap = cv2.VideoCapture(temp_video_path)
            temp_meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            temp_meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

        ordered_corners = camera.order_points(np.array(user_court_corners, dtype=np.float32))
        camera.calibrate(ordered_corners, temp_meta["width"], temp_meta["height"], focal_length)
    cached_data = utils.load_from_cache(cache_file_path)
    if not cached_data:
        print("--- CACHE MISS: Running all trackers and processors ---")

        action_cooldowns = {
            'serve': 120,
            'spike': 45,
            'block': 45,
            'set': 30,
            'defense': 30
        }

        ball_data = _get_ball_tracker().track_ball(temp_video_path, conf_thresh=0.6)
        action_classifications = _get_action_classifier().classify_action(temp_video_path, conf_thresh=0.2, sliding_window_size=3, action_cooldowns=action_cooldowns, trigger_count=1)
        player_data = _get_player_tracker().track_players(temp_video_path, conf_thresh=0.3, reid_sim_threshold=0.2)
        court_data = _get_court_tracker().track_court(temp_video_path, conf_thresh=0.3)
        serve_data = _get_serve_recognizer().recognize_serves(temp_video_path, court_corners=user_court_corners, conf_thresh=0.6, cooldown_frames=120)

        # We must calculate ball 3D positions before linking
        video_metadata = player_data.get("video_metadata", {})
        total_frames = video_metadata.get("total_frames", 0)
        ball_detections = [None] * total_frames
        for track in ball_data.get("ball_tracks", []):
            ball_detections[track['frame']] = track['bbox']
        
        ball_3d_positions = [None] * total_frames
        if camera.rvec is not None:
            for i, ball_bbox in enumerate(ball_detections):
                if ball_bbox:
                    ball_3d_positions[i] = camera.get_3d_position_estimation(ball_bbox, z_scale_calibration=calibration_params['z_scale'], x_sensitivity=calibration_params['x_sens'], ground_plane_offset=calibration_params['g_offset'])
        
        # --- MODIFIED CALL: Pass the calibrated camera object ---
        volleyball_events = utils.link_actions_to_players(
            action_classifications.get("action_detections", []),
            player_data.get("player_tracks", {}),
            ball_3d_positions,
            camera,
            calibration_params
        )

        # Consolidate all data for caching
        data_to_cache = {
            "video_metadata": video_metadata,
            "ball_data": ball_data,
            "action_classifications": action_classifications,
            "player_data": player_data,
            "serve_data": serve_data,
            "court_data": court_data,
            "volleyball_events": volleyball_events
        }
        
        utils.save_to_cache(cache_file_path, data_to_cache)
        cached_data = data_to_cache

    # --- 2. EXTRACT DATA (from cache or from fresh run) ---
    video_metadata = cached_data.get("video_metadata", {})
    ball_tracks = cached_data.get("ball_data", {}).get("ball_tracks", [])
    action_detections = cached_data.get("action_classifications", {}).get("action_detections", [])
    player_tracks = cached_data.get("player_data", {}).get("player_tracks", {})
    serve_events = cached_data.get("serve_data", {}).get("serve_events", [])
    total_frames = video_metadata.get("total_frames", 0)
    
    # Re-calculate ball 3D positions with user parameters
    ball_detections = [None] * total_frames
    for track in ball_tracks:
        if track['frame'] < total_frames:
            ball_detections[track['frame']] = track['bbox']
            
    ball_3d_positions = [None] * total_frames
    if camera.rvec is not None:
        for i, ball_bbox in enumerate(ball_detections):
            if ball_bbox:
                ball_3d_positions[i] = camera.get_3d_position_estimation(ball_bbox, z_scale_calibration=calibration_params['z_scale'], x_sensitivity=calibration_params['x_sens'], ground_plane_offset=calibration_params['g_offset'])
    
    volleyball_events = utils.link_actions_to_players(
        action_detections, player_tracks, ball_3d_positions, camera, calibration_params
    )
         
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    # --- 3. RETURN FINAL RESPONSE ---
    return {
        "video_metadata": video_metadata,
        "court_detections": [user_court_corners] * total_frames if user_court_corners else [],
        "ball_detections": ball_detections,
        "action_detections": action_detections,
        "serve_events": serve_events,
        "player_tracks": player_tracks,
        "ball_3d_positions": ball_3d_positions,
        "volleyball_events": volleyball_events
    }


@app.post("/track-ball/")
async def track_ball_endpoint(file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    temp_video_path = utils.save_temp_video(file)
    tracking_data = BallTracker(
        s3_bucket=s3_locations["ball_tracker"]["s3_bucket"],
        s3_key=s3_locations["ball_tracker"]["s3_key"]
    ).track_ball(temp_video_path)
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    return tracking_data


@app.post("/track-court/")
async def track_court_endpoint(file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    temp_video_path = utils.save_temp_video(file)
    tracking_data = CourtTracker(
        s3_bucket=s3_locations["court_tracker"]["s3_bucket"],
        s3_key=s3_locations["court_tracker"]["s3_key"]
    ).track_court(temp_video_path)
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    return tracking_data


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)