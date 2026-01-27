import cv2
import numpy as np
import random

class Camera:
    def __init__(self, camera_height_m=None):
        self.rvec = None
        self.tvec = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        self.camera_height_m = camera_height_m
        
        # --- FIX 1: Redefine the world coordinate system to be camera-friendly ---
        # Y is now the vertical "UP" axis. Z is now the depth axis.
        self.world_coords = np.array([
            [0, 0, 0],   # TL (X=0, Y/Height=0, Z/Depth=0)
            [9, 0, 0],   # TR (X=9, Y/Height=0, Z/Depth=0)
            [9, 0, 18],  # BR (X=9, Y/Height=0, Z/Depth=18)
            [0, 0, 18]   # BL (X=0, Y/Height=0, Z/Depth=18)
        ], dtype=np.float32)

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        y_sorted = pts[np.argsort(pts[:, 1]), :]
        top_points = y_sorted[:2, :]
        bottom_points = y_sorted[2:, :]
        top_points = top_points[np.argsort(top_points[:, 0]), :]
        tl, tr = top_points
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0]), :]
        bl, br = bottom_points
        return np.array([tl, tr, br, bl], dtype="float32")

    def calibrate(self, image_corners, image_width=1920, image_height=1080, focal_length_multiplier=1.2):
        if image_corners is None or len(image_corners) != 4:
            return False

        focal_length = image_width * focal_length_multiplier
        center_x = image_width / 2
        center_y = image_height / 2
        self.camera_matrix = np.array(
            [[focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]], dtype=np.float32
        )

        initial_rvec = np.array([np.pi, 0, 0], dtype=np.float32)
        initial_tvec = np.array([4.5, self.camera_height_m or 10.0, 9.0], dtype=np.float32)

        success, self.rvec, self.tvec = cv2.solvePnP(
            self.world_coords, image_corners, self.camera_matrix, self.dist_coeffs,
            rvec=initial_rvec, tvec=initial_tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return
        
        if self.camera_height_m is not None and self.camera_height_m > 0:
            R, _ = cv2.Rodrigues(self.rvec)
            cam_pos_world_initial = -np.dot(np.linalg.inv(R), self.tvec)
            
            # Height is now the Y-component (index 1)
            calculated_height = cam_pos_world_initial[1]
            print(f"DEBUG: Initial calculated camera height is {calculated_height:.2f}m")

            target_cam_pos_world = np.array([
                cam_pos_world_initial[0],
                self.camera_height_m, # Enforce height on the Y-axis
                cam_pos_world_initial[2]
            ]).reshape(3, 1)

            self.tvec = -np.dot(R, target_cam_pos_world)
        
        print(f"DEBUG: Final rvec: {self.rvec.flatten()}")
        print(f"DEBUG: Final tvec: {self.tvec.flatten()}")
        return True

    def get_3d_position_estimation(self, bbox, ball_real_diameter_m=0.21, z_scale_calibration=1.5, x_sensitivity=2.0, ground_plane_offset=0.0):
        fallback = (4.5, 9.0, 1.0)

        if self.rvec is None or self.camera_matrix is None or bbox is None:
            return fallback

        x_min, y_min, x_max, y_max = bbox
        ball_center_x = (x_min + x_max) / 2
        ball_center_y = (y_min + y_max) / 2
        pixel_diameter = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        
        if pixel_diameter < 5: 
            return fallback

        try:
            focal_length_x = self.camera_matrix[0, 0]
            distance_to_ball = (focal_length_x * ball_real_diameter_m) / pixel_diameter

            pixel_vec = np.array([[[ball_center_x, ball_center_y]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(pixel_vec, self.camera_matrix, self.dist_coeffs)
            px, py = undistorted[0][0]
            
            ball_pos_camera = np.array([
                px * distance_to_ball, py * distance_to_ball, distance_to_ball
            ]).reshape(3, 1)

            R, _ = cv2.Rodrigues(self.rvec)
            ball_pos_world = np.dot(np.linalg.inv(R), (ball_pos_camera - self.tvec))

            pos = ball_pos_world.flatten()
            
            # Un-swap the axes to match your application's expectation
            w_x = pos[0]
            w_y = pos[2] # World Y (Depth) comes from calculation's Z
            raw_height = pos[1] # World Z (Height) comes from calculation's Y

            # Apply final calibrations
            w_x = 4.5 + ((w_x - 4.5) * x_sensitivity)
            
            # --- FIX 3: New height calculation for sensitivity control ---
            # First, we apply the offset to set the 'zero' point.
            # Then, we scale the result to control sensitivity.
            w_z = (raw_height - ground_plane_offset) * z_scale_calibration

            # Clamp the final values
            w_x = max(-5.0, min(14.0, w_x))
            w_y = max(-5.0, min(23.0, w_y))
            w_z = max(0.0, min(15.0, w_z))

            return (w_x, w_y, w_z)

        except Exception as e:
            return fallback

    def get_point_3d_position(self, point_2d, reference_real_height_m, z_scale_calibration=1.5, x_sensitivity=2.0, ground_plane_offset=0.0):
        fallback = (4.5, 9.0, reference_real_height_m) # fallback

        if self.rvec is None or self.camera_matrix is None or point_2d is None:
            return fallback

        try:
            # Step 1: Back-project the 2D point into a 3D ray from the camera
            pixel_vec = np.array([[[point_2d[0], point_2d[1]]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(pixel_vec, self.camera_matrix, self.dist_coeffs)
            px, py = undistorted[0][0]

            # Step 2: Invert camera rotation to work in world coordinates
            R, _ = cv2.Rodrigues(self.rvec)
            R_inv = np.linalg.inv(R)
            
            # Step 3: Find camera position in world coordinates
            cam_pos_world = -np.dot(R_inv, self.tvec).flatten()

            # Step 4: Find the direction of the ray in world coordinates
            ray_dir_cam = np.array([px, py, 1.0])
            ray_dir_world = np.dot(R_inv, ray_dir_cam)
            
            # We assume the point is at the reference height. Find the intersection of the ray
            # with the horizontal plane at y = reference_real_height_m.
            # Ray equation: P = cam_pos_world + t * ray_dir_world
            # P.y = reference_real_height_m
            # cam_pos_world.y + t * ray_dir_world.y = reference_real_height_m
            
            if abs(ray_dir_world[1]) < 1e-6: # div by zero
                return fallback
                
            t = (reference_real_height_m - cam_pos_world[1]) / ray_dir_world[1]
            
            point_3d_world = cam_pos_world + t * ray_dir_world
            pos = point_3d_world.flatten()
            w_x = pos[0]
            w_y = pos[2] # World Y (Depth) comes from calculation's Z
            raw_height = pos[1] # World Z (Height) comes from calculation's Y

            # Apply final calibrations
            w_x = 4.5 + ((w_x - 4.5) * x_sensitivity)
            w_z = (raw_height - ground_plane_offset) * z_scale_calibration

            # Clamp the final values
            w_x = max(-5.0, min(14.0, w_x))
            w_y = max(-5.0, min(23.0, w_y))
            w_z = max(0.0, min(15.0, w_z))
            
            return (w_x, w_y, w_z)

        except Exception as e:
            return fallback