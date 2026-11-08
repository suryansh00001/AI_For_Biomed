"""
3D Rigid Body Geometry & Multi-Face Pose Estimation for the Surgical Cube Stylus.
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from surgical_tracking.config import SurgicalTrackingConfig


class SurgicalCubeModel:
    """
    Defines the 3D corner coordinates of all fiducial markers on the cube faces
    in the local Cube Coordinate Frame (origin at cube center).
    
    Axes convention:
      +X: Right,  -X: Left
      +Y: Top,    -Y: Bottom (pen shaft axis)
      +Z: Front,  -Z: Back
    """
    def __init__(self, config: Optional[SurgicalTrackingConfig] = None):
        if config is None:
            config = SurgicalTrackingConfig()
        self.config = config
        self.cube_size = config.cube_size_mm
        self.marker_size = config.marker_size_mm
        self.pen_length = config.pen_length_mm
        
        # Pen tip offset vector in local cube coordinates:
        # The dice sits at the top/rear of the pen. The pen shaft extends DOWNWARDS (+Y)
        # towards the operating table / patient.
        self.tip_offset = np.array([0.0, self.pen_length, 0.0], dtype=np.float64)
        
        # Build 3D corner points for each face
        self.face_3d_points = self._build_face_3d_points()
        
        # Inverted mapping: marker_id -> 3D corners array (4, 3)
        self.marker_id_to_3d: Dict[int, np.ndarray] = {}
        for face_name, marker_id in config.face_marker_ids.items():
            if face_name in self.face_3d_points:
                self.marker_id_to_3d[marker_id] = self.face_3d_points[face_name]

    def _build_face_3d_points(self) -> Dict[str, np.ndarray]:
        """
        Builds 3D corner coordinates (TL, TR, BR, BL) in millimeters for each face.
        """
        s = self.marker_size / 2.0
        h = self.cube_size / 2.0
        
        points = {
            # FRONT face (+Z normal): viewed facing +Z
            "FRONT": np.array([
                [-s,  s,  h],  # Top-Left
                [ s,  s,  h],  # Top-Right
                [ s, -s,  h],  # Bottom-Right
                [-s, -s,  h],  # Bottom-Left
            ], dtype=np.float64),
            
            # BACK face (-Z normal): viewed from back looking -Z
            "BACK": np.array([
                [ s,  s, -h],  # Top-Left
                [-s,  s, -h],  # Top-Right
                [-s, -s, -h],  # Bottom-Right
                [ s, -s, -h],  # Bottom-Left
            ], dtype=np.float64),
            
            # RIGHT face (+X normal): viewed from right looking -X
            "RIGHT": np.array([
                [ h,  s, -s],  # Top-Left
                [ h,  s,  s],  # Top-Right
                [ h, -s,  s],  # Bottom-Right
                [ h, -s, -s],  # Bottom-Left
            ], dtype=np.float64),
            
            # LEFT face (-X normal): viewed from left looking +X
            "LEFT": np.array([
                [-h,  s,  s],  # Top-Left
                [-h,  s, -s],  # Top-Right
                [-h, -s, -s],  # Bottom-Right
                [-h, -s,  s],  # Bottom-Left
            ], dtype=np.float64),
            
            # TOP face (+Y normal): viewed from top looking down -Y
            "TOP": np.array([
                [-s,  h, -s],  # Top-Left
                [ s,  h, -s],  # Top-Right
                [ s,  h,  s],  # Bottom-Right
                [-s,  h,  s],  # Bottom-Left
            ], dtype=np.float64),
            
            # BOTTOM face (-Y normal): (pen shaft attaches here)
            "BOTTOM": np.array([
                [-s, -h,  s],  # Top-Left
                [ s, -h,  s],  # Top-Right
                [ s, -h, -s],  # Bottom-Right
                [-s, -h, -s],  # Bottom-Left
            ], dtype=np.float64),
        }
        return points

    def estimate_cube_pose(
        self,
        corners: List[np.ndarray],
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Optional[Dict]:
        """
        Aggregates all visible marker corners belonging to the cube and performs
        joint 6-DOF solvePnP estimation.
        
        Returns dict containing:
          - rvec, tvec (cube pose in camera frame)
          - R (3x3 rotation matrix)
          - tip_pos_cam (3,) 3D coordinates of the stylus tip
          - axis_vector_cam (3,) unit vector pointing along stylus trajectory
          - num_faces_visible (int)
          - visible_marker_ids (list of int)
        """
        if ids is None or len(ids) == 0:
            return None
            
        obj_points = []
        img_points = []
        visible_ids = []
        
        flat_ids = ids.flatten()
        for i, mid in enumerate(flat_ids):
            if mid in self.marker_id_to_3d:
                # Shape (4, 3) 3D points
                pts_3d = self.marker_id_to_3d[mid]
                # Shape (4, 2) 2D pixel corners
                pts_2d = corners[i].reshape(-1, 2)
                
                obj_points.append(pts_3d)
                img_points.append(pts_2d)
                visible_ids.append(int(mid))
                
        if len(obj_points) == 0:
            return None
            
        all_obj_pts = np.vstack(obj_points).astype(np.float64)
        all_img_pts = np.vstack(img_points).astype(np.float64)
        
        # Need at least 4 coplanar points (1 marker) or more for solvePnP
        success, rvec, tvec = cv2.solvePnP(
            all_obj_pts,
            all_img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
            
        # Refine pose using Levenberg-Marquardt
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                all_obj_pts,
                all_img_pts,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec
            )
        except Exception:
            pass  # Fall back to initial estimate
            
        R, _ = cv2.Rodrigues(rvec)
        
        # Calculate physical tool tip in camera frame
        # P_tip = R * t_tip + T
        tip_cam = (R @ self.tip_offset) + tvec.flatten()
        
        # Tool pointing vector along pen shaft (+Y towards tip)
        unit_shaft = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis_cam = R @ unit_shaft
        axis_cam = axis_cam / np.linalg.norm(axis_cam)
        
        # Calculate Euler angles (Roll, Pitch, Yaw) in degrees
        euler_deg = self._rotation_matrix_to_euler_angles(R)
        
        return {
            "rvec": rvec,
            "tvec": tvec,
            "R": R,
            "euler_deg": euler_deg,
            "tip_pos_cam": tip_cam,
            "axis_vector_cam": axis_cam,
            "num_faces_visible": len(visible_ids),
            "visible_marker_ids": visible_ids
        }

    @staticmethod
    def _rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
        """Computes Roll, Pitch, Yaw in degrees from 3x3 rotation matrix."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees(np.array([x, y, z]))

