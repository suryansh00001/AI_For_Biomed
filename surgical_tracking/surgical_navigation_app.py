"""
Real-Time 6-DOF Surgical Tool Tracking & AR Brain Navigation Station.
Uses real 3D patient brain files (BraTS2020) directly from exported_3d_models/.
Tracks a multi-face fiducial cube stylus, computes 6-DOF pose & tool tip coordinates,
spawns the real 3D patient brain model in AR, and displays live surgical margin telemetry.
"""
import sys
import os
import time
import argparse
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from surgical_tracking.config import SurgicalTrackingConfig
from surgical_tracking.cube_model import SurgicalCubeModel
from surgical_tracking.filter import Pose6DOFFilter
from surgical_tracking.pivot_calibration import PivotCalibrator
from surgical_tracking.brain_ar_overlay import PatientARBrain
from surgical_tracking.streamer import SurgicalTelemetryStreamer


class SurgicalNavigationStation:
    def __init__(self, config: SurgicalTrackingConfig = None, initial_suffix: str = ""):
        if config is None:
            config = SurgicalTrackingConfig()
        self.config = config
        self.models_dir = os.path.join(PROJECT_ROOT, "exported_3d_models")
        
        # Discover all real project patients
        self.available_patients = PatientARBrain.get_available_patients(self.models_dir)
        if not self.available_patients:
            raise RuntimeError(f"No real patient 3D models found in {self.models_dir}!")
            
        if self.config.target_patient_id not in self.available_patients:
            self.config.target_patient_id = self.available_patients[0]
            
        self.patient_idx = self.available_patients.index(self.config.target_patient_id)
        
        # Segmentation modes: "" (Standard), "_model" (AI U-Net Predicted), "_gt" (Ground Truth)
        self.segmentation_modes = ["", "_model", "_gt"]
        self.mode_labels = {"": "Standard", "_model": "AI U-Net Predicted", "_gt": "Ground Truth"}
        self.current_mode_idx = self.segmentation_modes.index(initial_suffix) if initial_suffix in self.segmentation_modes else 0
        
        # Core tracking & AR modules
        self.cube_model = SurgicalCubeModel(config)
        self.pose_filter = Pose6DOFFilter(
            min_cutoff=config.one_euro_min_cutoff,
            beta=config.one_euro_beta
        )
        self.pivot_calibrator = PivotCalibrator()
        self.streamer = SurgicalTelemetryStreamer()
        
        # Real AR Brain instance
        self.ar_brain = PatientARBrain(
            patient_id=self.config.target_patient_id,
            models_dir=self.models_dir,
            suffix=self.segmentation_modes[self.current_mode_idx],
            scale_factor=self.config.brain_scale
        )
        
        # ArUco detector setup
        self.dictionary = cv2.aruco.getPredefinedDictionary(config.aruco_dict_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Display toggles
        self.show_cortex = True
        self.show_trajectory = True
        self.is_pivoting = False
        
        # Camera intrinsics (default initial guess if uncalibrated)
        w, h = config.frame_width, config.frame_height
        focal_length = w
        self.camera_matrix = np.array([
            [focal_length, 0.0, w / 2.0],
            [0.0, focal_length, h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def switch_patient(self, direction: int = 1):
        """Switches to the next/previous real patient in exported_3d_models."""
        self.patient_idx = (self.patient_idx + direction) % len(self.available_patients)
        self.config.target_patient_id = self.available_patients[self.patient_idx]
        mode = self.segmentation_modes[self.current_mode_idx]
        self.ar_brain.load_patient(self.config.target_patient_id, mode)

    def cycle_segmentation_mode(self):
        """Cycles between Standard, AI Predicted (_model), and Ground Truth (_gt)."""
        self.current_mode_idx = (self.current_mode_idx + 1) % len(self.segmentation_modes)
        mode = self.segmentation_modes[self.current_mode_idx]
        self.ar_brain.load_patient(self.config.target_patient_id, mode)

    def draw_cube_wireframe_and_axes(self, frame, rvec, tvec):
        """Draws a 3D bounding wireframe around the dice and RGB coordinate axes."""
        h = self.config.cube_size_mm / 2.0
        cube_corners = np.array([
            [-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h],
            [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h]
        ], dtype=np.float64)
        
        pts, _ = cv2.projectPoints(cube_corners, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        pts = pts.reshape(-1, 2).astype(np.int32)
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for e0, e1 in edges:
            cv2.line(frame, tuple(pts[e0]), tuple(pts[e1]), (200, 200, 200), 1, cv2.LINE_AA)
            
        axis_len = 25.0
        axes_3d = np.array([
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],  # X: Red
            [0.0, axis_len, 0.0],  # Y: Green
            [0.0, 0.0, axis_len]   # Z: Blue
        ], dtype=np.float64)
        ax_pts, _ = cv2.projectPoints(axes_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        ax_pts = ax_pts.reshape(-1, 2).astype(np.int32)
        
        origin = tuple(ax_pts[0])
        cv2.line(frame, origin, tuple(ax_pts[1]), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(frame, origin, tuple(ax_pts[2]), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, origin, tuple(ax_pts[3]), (255, 100, 0), 2, cv2.LINE_AA)

    def draw_stylus_tool(self, frame, rvec, tvec, tip_pos_cam, axis_cam):
        """Draws the virtual pen shaft, glowing tip reticle, and surgical trajectory ray."""
        R, _ = cv2.Rodrigues(rvec)
        T = tvec.flatten()
        
        # Pen attaches to bottom face (+Y) of the cube towards the patient
        shaft_start_local = np.array([0.0, self.config.cube_size_mm / 2.0, 0.0])
        shaft_start_cam = (R @ shaft_start_local) + T
        
        pts_3d = np.vstack([shaft_start_cam, tip_pos_cam])
        pts_2d, _ = cv2.projectPoints(
            pts_3d,
            np.zeros((3, 1)), np.zeros((3, 1)),
            self.camera_matrix, self.dist_coeffs
        )
        pts_2d = pts_2d.reshape(-1, 2).astype(np.int32)
        
        p_start = tuple(pts_2d[0])
        p_tip = tuple(pts_2d[1])
        
        # Stylized Surgical Stylus Shaft (Surgical Steel / Metallic look)
        cv2.line(frame, p_start, p_tip, (40, 40, 40), 9, cv2.LINE_AA)
        cv2.line(frame, p_start, p_tip, (160, 160, 165), 5, cv2.LINE_AA)
        cv2.line(frame, p_start, p_tip, (230, 235, 240), 2, cv2.LINE_AA)
        
        # Glowing Surgical Pointer Tip
        cv2.circle(frame, p_tip, 7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, p_tip, 3, (0, 0, 255), -1, cv2.LINE_AA)
        
        # Surgical Trajectory Laser Ray (projecting ahead of tip into target tissue)
        if self.show_trajectory:
            ray_end_cam = tip_pos_cam + (axis_cam * 110.0)
            ray_pts, _ = cv2.projectPoints(
                np.vstack([tip_pos_cam, ray_end_cam]),
                np.zeros((3, 1)), np.zeros((3, 1)),
                self.camera_matrix, self.dist_coeffs
            )
            ray_pts = ray_pts.reshape(-1, 2).astype(np.int32)
            cv2.line(frame, tuple(ray_pts[0]), tuple(ray_pts[1]), (0, 255, 140), 1, cv2.LINE_AA)
            cv2.circle(frame, tuple(ray_pts[1]), 3, (0, 255, 140), 1, cv2.LINE_AA)

    def draw_clinical_hud(self, frame, tracking_result, margins, fps):
        """Renders the surgical navigation telemetry panel."""
        h, w = frame.shape[:2]
        
        panel_w = 440
        panel_h = 340
        hud_bg = frame[20:20 + panel_h, 20:20 + panel_w].copy()
        dark = np.zeros_like(hud_bg)
        cv2.addWeighted(hud_bg, 0.20, dark, 0.80, 0, hud_bg)
        frame[20:20 + panel_h, 20:20 + panel_w] = hud_bg
        cv2.rectangle(frame, (20, 20), (20 + panel_w, 20 + panel_h), (70, 70, 70), 1)
        
        mode_str = self.mode_labels[self.segmentation_modes[self.current_mode_idx]]
        
        # Header
        cv2.putText(frame, "BRAIN SURGICAL NAVIGATION", (35, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        cv2.putText(frame, f"Patient: {self.config.target_patient_id} [{mode_str}]", (35, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:04.1f} | Scale: {self.ar_brain.scale_factor:.2f}x [+/-] | #{self.patient_idx + 1}/{len(self.available_patients)}",
                    (35, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)
        cv2.line(frame, (35, 96), (20 + panel_w - 15, 96), (80, 80, 80), 1)
        
        y = 118
        if tracking_result is not None:
            tip = tracking_result["tip_pos_cam"]
            euler = tracking_result["euler_deg"]
            faces = tracking_result["num_faces_visible"]
            ids = tracking_result["visible_marker_ids"]
            
            # Stylus Telemetry
            cv2.putText(frame, f"TOOL TIP (Cam mm): X:{tip[0]:+06.1f}  Y:{tip[1]:+06.1f}  Z:{tip[2]:+06.1f}",
                        (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
            y += 24
            cv2.putText(frame, f"ORIENTATION (deg): R:{euler[0]:+05.1f}  P:{euler[1]:+05.1f}  Y:{euler[2]:+05.1f}",
                        (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            y += 24
            cv2.putText(frame, f"CUBE TRACKING: {faces} face(s) locked [IDs: {ids}]",
                        (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            y += 26
            
            # Margins & Anatomical Proximity
            cv2.line(frame, (35, y - 6), (20 + panel_w - 15, y - 6), (80, 80, 80), 1)
            if margins:
                d_cortex = margins.get("cortex_mm", 999.0)
                d_edema = margins.get("edema", 999.0)
                d_enh = margins.get("enhancing", 999.0)
                d_nec = margins.get("necrotic", 999.0)
                min_tumor_d = min(d_edema, d_enh, d_nec)
                
                if min_tumor_d < 5.0:
                    status_text = "CRITICAL: TUMOR CONTACT / PENETRATION"
                    status_color = (0, 0, 255)
                elif min_tumor_d < self.config.safety_margin_mm:
                    status_text = f"WARNING: MARGIN < {self.config.safety_margin_mm}mm"
                    status_color = (0, 215, 255)
                else:
                    status_text = "STATUS: SAFE SURGICAL CORRIDOR"
                    status_color = (0, 255, 0)
                    
                cv2.putText(frame, status_text, (35, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color, 2)
                y += 34
                cv2.putText(frame, f"Brain Cortex Surface:    {d_cortex:05.1f} mm",
                            (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y += 20
                cv2.putText(frame, f"Edema Margin:            {d_edema:05.1f} mm",
                            (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 220, 255), 1)
                y += 20
                cv2.putText(frame, f"Enhancing Rim:           {d_enh:05.1f} mm",
                            (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 20), 1)
                y += 20
                cv2.putText(frame, f"Necrotic Core:           {d_nec:05.1f} mm",
                            (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 255), 1)
        else:
            cv2.putText(frame, "STYLUS CUBE: SEARCHING FOR MARKERS...",
                        (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
            y += 26
            cv2.putText(frame, "Rotate or bring stylus into camera view.",
                        (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
            
        if self.is_pivoting:
            banner = f"PIVOT CALIBRATING... Swivel pen ({len(self.pivot_calibrator.rotation_matrices)} samples)"
            cv2.putText(frame, banner, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            footer = "[N]: Patient | [M]: Mode | [C]: Cortex | [+/-]: Scale | [P]: Calib | [Q]: Exit"
            cv2.putText(frame, footer, (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    def run(self):
        """Main application loop."""
        print(f"[*] Initializing camera {self.config.camera_index}...")
        cap = cv2.VideoCapture(self.config.camera_index)
        
        if not cap.isOpened():
            print(f"[!] Unable to open camera {self.config.camera_index}.")
            print("[*] Launching in Synthetic Interactive Simulator Mode with REAL Brain Files...")
            self.run_simulation()
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        
        ret, sample_frame = cap.read()
        if ret and sample_frame is not None:
            fh, fw = sample_frame.shape[:2]
            focal = fw
            self.camera_matrix = np.array([
                [focal, 0.0, fw / 2.0],
                [0.0, focal, fh / 2.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            
        prev_time = time.time()
        print("\n[+] Real Brain Surgical Navigation Station active.")
        print("    Keys: [N] Next Patient | [M] AI/GT Mode | [C] Cortex | [T] Laser | [P] Pivot | [Q] Quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            curr_time = time.time()
            fps = 1.0 / max(1e-4, (curr_time - prev_time))
            prev_time = curr_time
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.detector.detectMarkers(gray)
            
            # Check for Patient Anchor Marker (ID 0)
            if ids is not None and len(ids) > 0:
                flat_ids = ids.flatten()
                for idx, mid in enumerate(flat_ids):
                    if mid == self.config.patient_anchor_id:
                        rvec_a, tvec_a, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [corners[idx]],
                            self.config.anchor_size_mm,
                            self.camera_matrix,
                            self.dist_coeffs
                        )
                        self.ar_brain.update_anchor_from_marker(rvec_a[0].reshape(3, 1), tvec_a[0].reshape(3, 1))
                        break
                        
            # Estimate 6-DOF Stylus Cube Pose
            tracking = self.cube_model.estimate_cube_pose(
                corners, ids, self.camera_matrix, self.dist_coeffs
            )
            
            margins = None
            if tracking is not None:
                sm_tvec, sm_rvec, sm_tip = self.pose_filter.filter_pose(
                    tracking["tvec"], tracking["rvec"], tracking["tip_pos_cam"], curr_time
                )
                tracking["tvec"] = sm_tvec
                tracking["rvec"] = sm_rvec
                tracking["tip_pos_cam"] = sm_tip
                
                R_sm, _ = cv2.Rodrigues(sm_rvec)
                tracking["R"] = R_sm
                tracking["axis_vector_cam"] = R_sm @ np.array([0.0, 1.0, 0.0])
                tracking["euler_deg"] = self.cube_model._rotation_matrix_to_euler_angles(R_sm)
                
                if self.is_pivoting:
                    self.pivot_calibrator.add_sample(R_sm, sm_tvec)
                    
                margins = self.ar_brain.calculate_tool_margins(sm_tip)

            # Render AR Holographic Real Brain & Tumors
            frame = self.ar_brain.render_ar_overlay(
                frame, self.camera_matrix, self.dist_coeffs,
                show_cortex=self.show_cortex
            )
            
            # Render Tracked Stylus Tool
            if tracking is not None:
                self.draw_cube_wireframe_and_axes(frame, tracking["rvec"], tracking["tvec"])
                self.draw_stylus_tool(
                    frame, tracking["rvec"], tracking["tvec"],
                    tracking["tip_pos_cam"], tracking["axis_vector_cam"]
                )
                
            # Render Clinical HUD
            self.draw_clinical_hud(frame, tracking, margins, fps)
            
            cv2.imshow("BraTS 6-DOF Surgical AR Navigation Station [REAL BRAIN]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('n') or key == ord(']'):
                self.switch_patient(1)
            elif key == ord('['):
                self.switch_patient(-1)
            elif key == ord('m'):
                self.cycle_segmentation_mode()
            elif key == ord('c'):
                self.show_cortex = not self.show_cortex
            elif key == ord('t'):
                self.show_trajectory = not self.show_trajectory
            elif key == ord('+') or key == ord('='):
                self.ar_brain.set_scale(min(3.0, self.ar_brain.scale_factor * 1.15))
            elif key == ord('-') or key == ord('_'):
                self.ar_brain.set_scale(max(0.05, self.ar_brain.scale_factor * 0.85))
            elif key == ord('p'):
                self.is_pivoting = not self.is_pivoting
                if self.is_pivoting:
                    print("[*] Pivot calibration STARTED. Swivel stylus tip in a stationary divot.")
                    self.pivot_calibrator.start()
                else:
                    success, res = self.pivot_calibrator.stop_and_solve()
                    if success:
                        print(f"[+] Calibration SUCCESS! Offset: {res['tip_offset']}, RMSE: {res['rmse_mm']:.2f}mm")
                        self.cube_model.tip_offset = res["tip_offset"]
                    else:
                        print(f"[!] Calibration failed: {res.get('error')}")

        cap.release()
        cv2.destroyAllWindows()

    def run_simulation(self):
        """Interactive Simulator with real project patient brain meshes."""
        w, h = self.config.frame_width, self.config.frame_height
        prev_time = time.time()
        t = 0.0
        
        print(f"\n[+] Interactive Simulation with REAL BRAIN ({self.config.target_patient_id}).")
        print("    Keys: [N] Next Patient | [M] AI/GT Mode | [C] Cortex | [T] Laser | [Q] Quit\n")
        
        while True:
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / max(1e-4, dt)
            prev_time = curr_time
            t += dt
            
            frame = np.ones((h, w, 3), dtype=np.uint8) * 18
            
            # Surgeon holding stylus above the brain and navigating towards tumor
            cube_x = np.sin(t * 0.7) * 40.0 + 10.0
            cube_y = -95.0 + np.sin(t * 0.5) * 15.0
            cube_z = 370.0 + np.cos(t * 0.4) * 25.0
            tvec = np.array([[cube_x], [cube_y], [cube_z]], dtype=np.float64)
            
            # Stylus angled pointing down-forward into the brain tissue
            pitch = 0.35 + np.sin(t * 0.4) * 0.10
            yaw = -0.22 + np.cos(t * 0.5) * 0.08
            roll = np.sin(t * 0.6) * 0.12
            rvec = np.array([[pitch], [yaw], [roll]], dtype=np.float64)
            R, _ = cv2.Rodrigues(rvec)
            
            tip_pos = (R @ self.cube_model.tip_offset) + tvec.flatten()
            axis_cam = R @ np.array([0.0, 1.0, 0.0])
            axis_cam = axis_cam / np.linalg.norm(axis_cam)
            euler = self.cube_model._rotation_matrix_to_euler_angles(R)
            
            sim_tracking = {
                "rvec": rvec,
                "tvec": tvec,
                "R": R,
                "tip_pos_cam": tip_pos,
                "axis_vector_cam": axis_cam,
                "euler_deg": euler,
                "num_faces_visible": 2,
                "visible_marker_ids": [1, 4]
            }
            
            margins = self.ar_brain.calculate_tool_margins(tip_pos)
            
            frame = self.ar_brain.render_ar_overlay(
                frame, self.camera_matrix, self.dist_coeffs,
                show_cortex=self.show_cortex
            )
            
            self.draw_cube_wireframe_and_axes(frame, rvec, tvec)
            self.draw_stylus_tool(frame, rvec, tvec, tip_pos, axis_cam)
            self.draw_clinical_hud(frame, sim_tracking, margins, fps)
            
            cv2.imshow("BraTS 6-DOF Surgical AR Navigation Station [REAL BRAIN SIM]", frame)
            key = cv2.waitKey(16) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('n') or key == ord(']'):
                self.switch_patient(1)
            elif key == ord('['):
                self.switch_patient(-1)
            elif key == ord('m'):
                self.cycle_segmentation_mode()
            elif key == ord('c'):
                self.show_cortex = not self.show_cortex
            elif key == ord('t'):
                self.show_trajectory = not self.show_trajectory
            elif key == ord('+') or key == ord('='):
                self.ar_brain.set_scale(min(3.0, self.ar_brain.scale_factor * 1.15))
            elif key == ord('-') or key == ord('_'):
                self.ar_brain.set_scale(max(0.05, self.ar_brain.scale_factor * 0.85))

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6-DOF Surgical AR Navigation Station (Real Brain Models)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--patient", type=str, default="BraTS20_Training_001", help="Patient ID")
    parser.add_argument("--mode", type=str, default="", choices=["", "_model", "_gt"], help="Segmentation mode: '' (Standard), '_model' (AI U-Net), '_gt' (Ground Truth)")
    parser.add_argument("--sim", action="store_true", help="Force synthetic simulation mode")
    parser.add_argument("--cube-size", type=float, default=25.0, help="Cube edge length in mm")
    parser.add_argument("--marker-size", type=float, default=20.0, help="Marker edge length in mm")
    parser.add_argument("--anchor-size", type=float, default=75.0, help="Physical size of table anchor marker ID 0 in mm")
    parser.add_argument("--pen-length", type=float, default=135.0, help="Distance from cube center to tip in mm")
    parser.add_argument("--brain-scale", type=float, default=0.28, help="Scale factor for the AR brain (default: 0.28, 3.5x reduced)")
    args = parser.parse_args()
    
    cfg = SurgicalTrackingConfig(
        camera_index=args.camera,
        target_patient_id=args.patient,
        cube_size_mm=args.cube_size,
        marker_size_mm=args.marker_size,
        anchor_size_mm=args.anchor_size,
        pen_length_mm=args.pen_length,
        brain_scale=args.brain_scale
    )
    
    station = SurgicalNavigationStation(cfg, initial_suffix=args.mode)
    if args.sim:
        station.run_simulation()
    else:
        station.run()
