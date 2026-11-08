"""
Marker Generator for Surgical Stylus Cube & AR Patient Anchor.
Generates printable ArUco markers and a ready-to-cut cube net template.
"""
import os
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from surgical_tracking.config import SurgicalTrackingConfig


def create_marker_image(marker_id: int, dict_id: int, pixel_size: int = 400) -> np.ndarray:
    """Generates an ArUco marker image with a white border."""
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, pixel_size)
    
    # Add quiet border
    border = pixel_size // 8
    bordered = cv2.copyMakeBorder(
        marker, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255
    )
    return bordered


def generate_marker_suite(output_dir: str = "surgical_tracking/printable_markers",
                          config: SurgicalTrackingConfig = None):
    """Generates individual marker files and a composite printable net."""
    if config is None:
        config = SurgicalTrackingConfig()
        
    os.makedirs(output_dir, exist_ok=True)
    dict_id = config.aruco_dict_id
    
    print(f"[*] Generating ArUco markers for dictionary {dict_id}...")
    
    # 1. Patient Anchor Marker (ID 0)
    anchor_img = create_marker_image(config.patient_anchor_id, dict_id, pixel_size=600)
    anchor_path = os.path.join(output_dir, f"patient_anchor_id_{config.patient_anchor_id}.png")
    cv2.imwrite(anchor_path, anchor_img)
    print(f"  -> Saved patient anchor marker: {anchor_path}")
    
    # 2. Individual Cube Face Markers
    face_images = {}
    for face_name, marker_id in config.face_marker_ids.items():
        m_img = create_marker_image(marker_id, dict_id, pixel_size=400)
        face_images[face_name] = m_img
        out_path = os.path.join(output_dir, f"cube_marker_{face_name.lower()}_id_{marker_id}.png")
        cv2.imwrite(out_path, m_img)
        print(f"  -> Saved {face_name} marker (ID {marker_id}): {out_path}")
        
    # 3. Create a Cube Unfolded "Cross Net" Sheet
    # Layout of cube unfolding:
    #       [ TOP ]
    # [LEFT][FRONT][RIGHT][BACK]
    #       [BOTTOM]
    cell_px = 300
    net_width = cell_px * 4 + 100
    net_height = cell_px * 3 + 180
    
    sheet = np.ones((net_height, net_width, 3), dtype=np.uint8) * 255
    
    # Placement map on the 3x4 grid: (row, col)
    # row 0, col 1: TOP
    # row 1, col 0: LEFT, col 1: FRONT, col 2: RIGHT, col 3: BACK
    # row 2, col 1: BOTTOM
    placements = {
        "TOP": (0, 1),
        "LEFT": (1, 0),
        "FRONT": (1, 1),
        "RIGHT": (1, 2),
        "BACK": (1, 3),
        "BOTTOM": (2, 1)
    }
    
    offset_x = 50
    offset_y = 60
    
    for face_name, (r, c) in placements.items():
        marker_id = config.face_marker_ids[face_name]
        m_raw = face_images[face_name]
        m_resized = cv2.resize(m_raw, (cell_px - 20, cell_px - 20))
        m_bgr = cv2.cvtColor(m_resized, cv2.COLOR_GRAY2BGR)
        
        px = offset_x + c * cell_px
        py = offset_y + r * cell_px
        
        # Paste marker inside cell
        sheet[py + 10:py + 10 + m_resized.shape[0], px + 10:px + 10 + m_resized.shape[1]] = m_bgr
        
        # Draw solid cut/fold outline around the face
        cv2.rectangle(sheet, (px, py), (px + cell_px, py + cell_px), (180, 180, 180), 2)
        
    # Header instructions
    title = "Surgical Stylus Fiducial Cube Net (Print & Fold onto Dice)"
    subtitle = f"Cube edge: {config.cube_size_mm}mm | Marker: {config.marker_size_mm}mm | Pen shaft attaches at BOTTOM"
    cv2.putText(sheet, title, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
    cv2.putText(sheet, subtitle, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    
    net_path = os.path.join(output_dir, "printable_cube_net_template.png")
    cv2.imwrite(net_path, sheet)
    print(f"[+] Successfully generated Printable Cube Net: {net_path}")
    return net_path


if __name__ == "__main__":
    generate_marker_suite()

