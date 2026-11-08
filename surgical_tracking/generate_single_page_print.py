"""
Single-Page High-Resolution Printable Sheet Generator.
Combines the 6-DOF Stylus Cube Net, the Patient Anchor Marker (ID 0),
a millimeter calibration ruler, and assembly instructions onto a single
standard A4 / Letter printable page (PNG & PDF).
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from surgical_tracking.config import SurgicalTrackingConfig


def generate_single_page_sheet(output_dir: str = "surgical_tracking/printable_markers",
                               config: SurgicalTrackingConfig = None):
    if config is None:
        config = SurgicalTrackingConfig()
        
    os.makedirs(output_dir, exist_ok=True)
    dict_id = config.aruco_dict_id
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    
    # Standard A4 at 300 DPI: 2480 x 3508 pixels
    # 1 mm ≈ 11.81 pixels
    DPI = 300
    MM_TO_PX = DPI / 25.4  # ~11.811 px per mm
    
    page_w = int(210 * MM_TO_PX)  # 2480 px
    page_h = int(297 * MM_TO_PX)  # 3508 px
    
    # White canvas
    sheet = np.ones((page_h, page_w, 3), dtype=np.uint8) * 255
    
    def mm(val):
        return int(val * MM_TO_PX)
        
    # ---------------------------------------------------------
    # 1. Header & Calibration Bar
    # ---------------------------------------------------------
    cv2.putText(sheet, "BRAIN SURGICAL NAVIGATION - ALL-IN-ONE PRINT SHEET",
                (mm(15), mm(16)), cv2.FONT_HERSHEY_DUPLEX, 1.3, (20, 20, 20), 3)
    cv2.putText(sheet, "Single Page: 6-DOF Stylus Cube Net + Patient Anchor Marker | Print at 100% scale (No fit-to-page)",
                (mm(15), mm(24)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (90, 90, 90), 2)
    
    # 50mm Calibration Ruler Check Bar
    ruler_x = mm(15)
    ruler_y = mm(30)
    ruler_w = mm(50)  # Exactly 50mm
    ruler_h = mm(6)
    cv2.rectangle(sheet, (ruler_x, ruler_y), (ruler_x + ruler_w, ruler_y + ruler_h), (0, 0, 0), 2)
    # 10mm tick marks
    for tick in range(6):
        tx = ruler_x + mm(tick * 10)
        cv2.line(sheet, (tx, ruler_y), (tx, ruler_y + ruler_h), (0, 0, 0), 2)
    cv2.putText(sheet, "<- Exactly 50 mm Ruler Scale Check ->",
                (ruler_x + mm(55), ruler_y + mm(5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
                
    cv2.line(sheet, (mm(15), mm(42)), (page_w - mm(15), mm(42)), (180, 180, 180), 2)

    # ---------------------------------------------------------
    # 2. Section A: Stylus Dice Folding Net (Top/Middle Area)
    # ---------------------------------------------------------
    sec_a_y = mm(50)
    cv2.putText(sheet, "PART 1: 6-DOF SURGICAL STYLUS CUBE NET (Fold onto Dice)",
                (mm(15), sec_a_y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 100, 200), 2)
    cv2.putText(sheet, f"Cube Edge: {config.cube_size_mm}mm | Marker Size: {config.marker_size_mm}mm | Pen attaches at BOTTOM face",
                (mm(15), sec_a_y + mm(7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 2)

    # Cube Net Layout (3 rows x 4 cols grid of cube faces):
    #       [TOP]
    # [LEFT][FRONT][RIGHT][BACK]
    #       [BOTTOM]
    cell_size = mm(config.cube_size_mm)
    marker_px = mm(config.marker_size_mm)
    
    net_start_x = mm(25)
    net_start_y = sec_a_y + mm(15)
    
    placements = {
        "TOP":    (0, 1, config.face_marker_ids["TOP"]),
        "LEFT":   (1, 0, config.face_marker_ids["LEFT"]),
        "FRONT":  (1, 1, config.face_marker_ids["FRONT"]),
        "RIGHT":  (1, 2, config.face_marker_ids["RIGHT"]),
        "BACK":   (1, 3, config.face_marker_ids["BACK"]),
        "BOTTOM": (2, 1, config.face_marker_ids["BOTTOM"]),
    }
    
    for face_name, (row, col, mid) in placements.items():
        fx = net_start_x + col * cell_size
        fy = net_start_y + row * cell_size
        
        # Solid border around each face (cut line)
        cv2.rectangle(sheet, (fx, fy), (fx + cell_size, fy + cell_size), (40, 40, 40), 2)
        
        # Generate ArUco marker image
        raw_marker = cv2.aruco.generateImageMarker(dictionary, mid, marker_px)
        marker_bgr = cv2.cvtColor(raw_marker, cv2.COLOR_GRAY2BGR)
        
        # Center marker inside cell (pristine black & white, no text overlay)
        offset = (cell_size - marker_px) // 2
        sheet[fy + offset:fy + offset + marker_px, fx + offset:fx + offset + marker_px] = marker_bgr

    # Instructions & Face Map beside the net (placed outside cut area)
    info_x = net_start_x + cell_size * 4 + mm(12)
    info_y = net_start_y + mm(6)
    cv2.putText(sheet, "Face ID Reference Map:", (info_x, info_y), cv2.FONT_HERSHEY_DUPLEX, 0.75, (20, 20, 20), 2)
    info_y += mm(7)
    
    face_map_lines = [
        "  * Face TOP    -> Marker ID 5",
        "  * Face FRONT  -> Marker ID 1",
        "  * Face RIGHT  -> Marker ID 4",
        "  * Face BACK   -> Marker ID 2",
        "  * Face LEFT   -> Marker ID 3",
        "  * Face BOTTOM -> Marker ID 6 (Pen shaft)",
    ]
    for line in face_map_lines:
        cv2.putText(sheet, line, (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 80, 180), 1)
        info_y += mm(5.5)

    info_y += mm(4)
    cv2.putText(sheet, "How to Assemble:", (info_x, info_y), cv2.FONT_HERSHEY_DUPLEX, 0.75, (20, 20, 20), 2)
    info_y += mm(7)
    
    instructions = [
        "1. Cut around outer cross boundary.",
        "2. Fold along interior square seams.",
        "3. Wrap & glue/tape firmly around dice.",
        "4. Tape or hot-glue pen to BOTTOM (ID 6).",
        "5. The pen now tracks continuously in 360!",
    ]
    for line in instructions:
        cv2.putText(sheet, line, (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 50, 50), 1)
        info_y += mm(5.5)

    # ---------------------------------------------------------
    # 3. Divider
    # ---------------------------------------------------------
    div_y = net_start_y + cell_size * 3 + mm(18)
    cv2.line(sheet, (mm(15), div_y), (page_w - mm(15), div_y), (180, 180, 180), 2)

    # ---------------------------------------------------------
    # 4. Section B: Patient Operating Table Anchor Marker (ID 0)
    # ---------------------------------------------------------
    sec_b_y = div_y + mm(12)
    cv2.putText(sheet, "PART 2: OPERATING TABLE ANCHOR MARKER (Spawns the AR Brain)",
                (mm(15), sec_b_y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 140, 80), 2)
    cv2.putText(sheet, "Marker ID: 0 | Place this flat on your desk/mousepad where the 3D Brain will appear in AR",
                (mm(15), sec_b_y + mm(7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 2)

    # Large 75mm x 75mm anchor marker
    anchor_size_mm = 75.0
    anchor_px = mm(anchor_size_mm)
    raw_anchor = cv2.aruco.generateImageMarker(dictionary, config.patient_anchor_id, anchor_px)
    anchor_bgr = cv2.cvtColor(raw_anchor, cv2.COLOR_GRAY2BGR)
    
    ax = mm(25)
    ay = sec_b_y + mm(16)
    
    # Outer cut border with quiet zone
    border = mm(6)
    cv2.rectangle(sheet, (ax - border, ay - border),
                  (ax + anchor_px + border, ay + anchor_px + border), (180, 180, 180), 2)
    sheet[ay:ay + anchor_px, ax:ax + anchor_px] = anchor_bgr
    
    # Anchor marker details beside it
    ax_info = ax + anchor_px + mm(18)
    ay_info = ay + mm(12)
    anchor_notes = [
        "Patient World Anchor Guide:",
        "* Size: 75mm high-contrast fiducial.",
        "* Place flat on your desk in view of the webcam.",
        "* The 3D patient brain model automatically locks",
        "  and spawns directly above this marker.",
        "* If marker is not in view, the station smoothly",
        "  anchors the brain at a calibrated spatial target.",
    ]
    for line in anchor_notes:
        cv2.putText(sheet, line, (ax_info, ay_info), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
        ay_info += mm(9)

    # ---------------------------------------------------------
    # 5. Footer Run Command
    # ---------------------------------------------------------
    cv2.line(sheet, (mm(15), page_h - mm(32)), (page_w - mm(15), page_h - mm(32)), (180, 180, 180), 2)
    cv2.putText(sheet, "LAUNCH COMMAND: python -m surgical_tracking.surgical_navigation_app --camera 0",
                (mm(15), page_h - mm(20)), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 80, 180), 2)
    cv2.putText(sheet, "Hotkeys: [N] Next Patient | [M] AI vs GT Model | [C] Toggle Cortex Shell | [P] Pivot Calibrate | [Q] Quit",
                (mm(15), page_h - mm(12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

    # Save PNG
    png_path = os.path.join(output_dir, "surgical_navigation_all_in_one_sheet.png")
    cv2.imwrite(png_path, sheet)
    print(f"[+] Successfully generated High-Res Print Sheet: {png_path}")
    
    # Also save as ready-to-print PDF
    pdf_path = os.path.join(output_dir, "surgical_navigation_all_in_one_sheet.pdf")
    pil_img = Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
    pil_img.save(pdf_path, "PDF", resolution=DPI)
    print(f"[+] Successfully generated Ready-to-Print PDF: {pdf_path}")
    
    return png_path, pdf_path


if __name__ == "__main__":
    generate_single_page_sheet()

