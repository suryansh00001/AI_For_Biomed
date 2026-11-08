"""
Configuration for 6-DOF Surgical Tool Tracking & AR Brain Navigation.
All spatial dimensions are in millimeters (mm) by default.
"""
from dataclasses import dataclass, field
from typing import Dict
import cv2


@dataclass
class SurgicalTrackingConfig:
    # Physical dimensions (mm)
    cube_size_mm: float = 25.0          # Edge length of the dice / cube
    marker_size_mm: float = 20.0        # Printed marker black-border edge length on cube
    anchor_size_mm: float = 75.0        # Physical printed size of Table Anchor Marker (ID 0)
    pen_length_mm: float = 135.0        # Vector from cube center to pen tip along +Y axis
    brain_scale: float = 0.28           # 3.5x reduction factor for AR brain model size
    
    # Fiducial Dictionary & IDs
    aruco_dict_id: int = cv2.aruco.DICT_4X4_50
    patient_anchor_id: int = 0          # Reference marker to spawn & pin the AR brain
    
    # Cube Face to Marker ID mapping
    # Cube coordinate frame: Origin (0,0,0) at cube center.
    # +X: Right, -X: Left
    # +Y: Top,   -Y: Bottom
    # +Z: Front, -Z: Back (Pen points along -Y or -Z)
    face_marker_ids: Dict[str, int] = field(default_factory=lambda: {
        "FRONT": 1,   # Face normal: +Z
        "BACK": 2,    # Face normal: -Z
        "LEFT": 3,    # Face normal: -X
        "RIGHT": 4,   # Face normal: +X
        "TOP": 5,     # Face normal: +Y
        "BOTTOM": 6   # Face normal: -Y (where pen shaft attaches)
    })
    
    # Camera settings
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    
    # Filter & Smoothing
    one_euro_min_cutoff: float = 1.0     # Reduce jitter when stationary
    one_euro_beta: float = 0.05          # Low latency when moving fast
    
    # AR Navigation parameters
    safety_margin_mm: float = 10.0       # Proximity warning radius
    target_patient_id: str = "BraTS20_Training_001"

