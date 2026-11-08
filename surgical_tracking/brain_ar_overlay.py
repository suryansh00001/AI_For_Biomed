"""
3D Brain & Multi-Compartment Tumor AR Rendering Engine.
Uses topological quadric decimation and a Painter's Algorithm Depth-Sorted
Shaded Surface Renderer to project solid 3D anatomical brain and tumor surfaces
onto the camera video feed with directional lighting and transparency.
"""
import os
import json
import cv2
import numpy as np
import trimesh
from typing import Dict, Optional, Tuple, List


class PatientARBrain:
    def __init__(self,
                 patient_id: str = "BraTS20_Training_001",
                 models_dir: str = "exported_3d_models",
                 suffix: str = "",
                 scale_factor: float = 1.0):
        self.models_dir = os.path.abspath(models_dir)
        self.patient_id = patient_id
        self.suffix = suffix
        self.scale_factor = scale_factor
        
        # Spatial pose in camera coordinates: centered 420mm in front of lens
        self.anchor_rvec = np.array([[0.2], [0.3], [0.0]], dtype=np.float64)
        self.anchor_tvec = np.array([[0.0], [10.0], [420.0]], dtype=np.float64)
        self.is_pinned_to_marker = False
        
        self.manifest_data = {}
        self.cortex_mesh: Optional[trimesh.Trimesh] = None
        self.cortex_full_vertices: Optional[np.ndarray] = None
        self.tumor_meshes: Dict[str, Dict] = {}
        
        self.load_patient(patient_id, suffix)

    @classmethod
    def get_available_patients(cls, models_dir: str = "exported_3d_models") -> List[str]:
        abs_dir = os.path.abspath(models_dir)
        if not os.path.exists(abs_dir):
            return []
        return [
            d for d in sorted(os.listdir(abs_dir))
            if os.path.isdir(os.path.join(abs_dir, d)) and d.startswith("BraTS20_")
        ]

    def load_patient(self, patient_id: str, suffix: str = ""):
        self.patient_id = patient_id
        self.suffix = suffix
        patient_dir = os.path.join(self.models_dir, patient_id)
        
        if not os.path.exists(patient_dir):
            raise FileNotFoundError(f"Real patient directory not found: {patient_dir}")
            
        print(f"\n[+] Loading real anatomical models: {patient_id} [{suffix or 'standard'}]")
        
        # Load clinical manifest
        manifest_filename = f"manifest{suffix}.json" if suffix else "manifest.json"
        manifest_path = os.path.join(patient_dir, manifest_filename)
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join(patient_dir, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    self.manifest_data = json.load(f)
            except Exception:
                pass

        # 1. Load & Decimate Real Brain Cortex
        cortex_file = f"brain_cortex{suffix}.obj" if suffix else "brain_cortex.obj"
        cortex_path = os.path.join(patient_dir, cortex_file)
        if not os.path.exists(cortex_path):
            cortex_path = os.path.join(patient_dir, "brain_cortex.obj")
            
        raw_cortex = trimesh.load(cortex_path)
        self.cortex_full_vertices = raw_cortex.vertices * self.scale_factor
        
        # Proper quadric decimation to preserve smooth continuous surface manifold
        try:
            # Reduce to ~700 faces for 60 FPS real-time depth sorting
            reduction = max(0.01, min(0.96, 1.0 - (750.0 / max(1, len(raw_cortex.faces)))))
            self.cortex_mesh = raw_cortex.simplify_quadric_decimation(reduction)
            self.cortex_mesh.vertices *= self.scale_factor
        except Exception:
            self.cortex_mesh = raw_cortex
            
        print(f"  -> Real Cortex: {len(self.cortex_mesh.faces)} surface faces.")

        # 2. Load Real Tumor Sub-Regions
        tumor_configs = {
            "edema": {
                "filename": f"tumor_edema{suffix}.obj",
                "fallback": "tumor_edema.obj",
                "color": (50, 220, 240),    # Emerald-Cyan BGR
                "alpha": 0.40,
                "label": "Edema Margin"
            },
            "enhancing": {
                "filename": f"tumor_enhancing{suffix}.obj",
                "fallback": "tumor_enhancing.obj",
                "color": (255, 160, 20),    # Bright Amber-Cyan BGR
                "alpha": 0.75,
                "label": "Enhancing Rim"
            },
            "necrotic": {
                "filename": f"tumor_necrotic{suffix}.obj",
                "fallback": "tumor_necrotic.obj",
                "color": (40, 40, 240),     # Rose-Red BGR
                "alpha": 0.90,
                "label": "Necrotic Core"
            }
        }
        
        self.tumor_meshes.clear()
        for key, conf in tumor_configs.items():
            tpath = os.path.join(patient_dir, conf["filename"])
            if not os.path.exists(tpath):
                tpath = os.path.join(patient_dir, conf["fallback"])
                
            if os.path.exists(tpath):
                try:
                    raw_t = trimesh.load(tpath)
                    reduction = max(0.01, min(0.92, 1.0 - (400.0 / max(1, len(raw_t.faces)))))
                    simplified_t = raw_t.simplify_quadric_decimation(reduction)
                    simplified_t.vertices *= self.scale_factor
                    self.tumor_meshes[key] = {
                        "mesh": simplified_t,
                        "color": conf["color"],
                        "alpha": conf["alpha"],
                        "label": conf["label"],
                        "full_vertices": raw_t.vertices * self.scale_factor
                    }
                    print(f"  -> Real Tumor [{conf['label']}]: {len(simplified_t.faces)} surface faces.")
                except Exception as e:
                    print(f"[!] Warning reading {tpath}: {e}")

    def set_scale(self, new_scale: float):
        """Dynamically rescales the 3D brain and internal tumor models."""
        old_scale = self.scale_factor
        if old_scale <= 0 or new_scale <= 0:
            return
        ratio = new_scale / old_scale
        self.scale_factor = new_scale
        if self.cortex_full_vertices is not None:
            self.cortex_full_vertices *= ratio
        if self.cortex_mesh is not None:
            self.cortex_mesh.vertices *= ratio
        for key, data in self.tumor_meshes.items():
            if data.get("mesh") is not None:
                data["mesh"].vertices *= ratio
            if data.get("full_vertices") is not None:
                data["full_vertices"] *= ratio
        print(f"[+] Brain scale updated to {self.scale_factor:.3f}x")

    def update_anchor_from_marker(self, rvec: np.ndarray, tvec: np.ndarray):
        """Pins the brain model in space directly above the Patient Anchor Marker (ID 0)."""
        R, _ = cv2.Rodrigues(rvec)
        # Position the brain comfortably above the table marker according to scale
        hover_height = max(15.0, 60.0 * self.scale_factor)
        elevation_local = np.array([0.0, 0.0, hover_height])
        elevated_tvec = tvec.flatten() + (R @ elevation_local)
        self.anchor_rvec = rvec.copy()
        self.anchor_tvec = elevated_tvec.reshape(3, 1)
        self.is_pinned_to_marker = True

    def calculate_tool_margins(self, tip_pos_cam: np.ndarray) -> Dict[str, float]:
        """Calculates exact Euclidean distance (mm) from tool tip to real tumor surfaces."""
        R, _ = cv2.Rodrigues(self.anchor_rvec)
        T = self.anchor_tvec.flatten()
        tip_local = R.T @ (tip_pos_cam.flatten() - T)
        
        distances = {}
        if self.cortex_full_vertices is not None and len(self.cortex_full_vertices) > 0:
            d_cortex = np.min(np.linalg.norm(self.cortex_full_vertices - tip_local, axis=1))
            distances["cortex_mm"] = float(d_cortex)
            
        for key, data in self.tumor_meshes.items():
            full_v = data["full_vertices"]
            if len(full_v) > 0:
                d_tumor = np.min(np.linalg.norm(full_v - tip_local, axis=1))
                distances[key] = float(d_tumor)
                
        return distances

    def _render_shaded_mesh(self,
                            frame: np.ndarray,
                            mesh: trimesh.Trimesh,
                            camera_matrix: np.ndarray,
                            dist_coeffs: np.ndarray,
                            base_color: Tuple[int, int, int],
                            alpha: float = 0.5,
                            draw_wire: bool = True):
        """
        Renders a solid 3D shaded surface with directional lighting,
        backface culling, and depth sorting (Painter's algorithm).
        """
        if mesh is None or len(mesh.faces) == 0:
            return
            
        R, _ = cv2.Rodrigues(self.anchor_rvec)
        T = self.anchor_tvec.flatten()
        
        # 3D vertices in camera coordinate space
        v_cam = (R @ mesh.vertices.T).T + T
        
        # Project 3D vertices to 2D image coordinates
        pts_2d, _ = cv2.projectPoints(
            mesh.vertices.astype(np.float64),
            self.anchor_rvec, self.anchor_tvec,
            camera_matrix, dist_coeffs
        )
        pts_2d = pts_2d.reshape(-1, 2).astype(np.int32)
        
        faces = mesh.faces
        f_pts_cam = v_cam[faces] # (N, 3, 3)
        centroids_z = np.mean(f_pts_cam[:, :, 2], axis=1)
        
        # Compute face normals in camera coordinate frame
        v0 = f_pts_cam[:, 0]
        v1 = f_pts_cam[:, 1]
        v2 = f_pts_cam[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norm_lens = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6
        normals = normals / norm_lens
        
        # View vectors (from face to camera origin [0,0,0])
        view_dirs = -v0
        view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-6)
        
        # Backface culling: discard faces pointing away from the camera
        dot_view = np.sum(normals * view_dirs, axis=1)
        front_faces = np.where(dot_view > -0.05)[0]
        if len(front_faces) == 0:
            return
            
        # Sort front faces back-to-front (largest Z first)
        sort_indices = front_faces[np.argsort(-centroids_z[front_faces])]
        
        # Clinical surgical overhead directional light
        light_dir = np.array([0.2, 0.5, 0.85], dtype=np.float64)
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.maximum(0.25, np.sum(normals * light_dir, axis=1))
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        b, g, r = base_color
        
        for fi in sort_indices:
            tri = pts_2d[faces[fi]]
            # Clip bounds
            if np.all(tri[:, 0] < 0) or np.all(tri[:, 0] >= w) or np.all(tri[:, 1] < 0) or np.all(tri[:, 1] >= h):
                continue
                
            intensity = diffuse[fi]
            shaded_color = (
                int(min(255, b * intensity)),
                int(min(255, g * intensity)),
                int(min(255, r * intensity))
            )
            cv2.fillConvexPoly(overlay, tri, shaded_color, cv2.LINE_AA)
            if draw_wire:
                wire_col = (
                    int(min(255, b * (intensity * 0.7 + 0.3))),
                    int(min(255, g * (intensity * 0.7 + 0.3))),
                    int(min(255, r * (intensity * 0.7 + 0.3)))
                )
                cv2.polylines(overlay, [tri], True, wire_col, 1, cv2.LINE_AA)
                
        # Alpha blend with background frame
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def render_ar_overlay(self,
                          frame: np.ndarray,
                          camera_matrix: np.ndarray,
                          dist_coeffs: np.ndarray,
                          show_cortex: bool = True,
                          cortex_opacity: float = 0.32) -> np.ndarray:
        """Renders solid 3D brain and internal tumor cores with depth ordering."""
        if self.anchor_rvec is None or self.anchor_tvec is None:
            return frame
            
        # 1. Render Internal Tumor Sub-Regions (Solid Glowing Volumes)
        # Render Edema (outermost tumor layer)
        if "edema" in self.tumor_meshes:
            t = self.tumor_meshes["edema"]
            self._render_shaded_mesh(
                frame, t["mesh"], camera_matrix, dist_coeffs,
                base_color=t["color"], alpha=t["alpha"], draw_wire=True
            )
            
        # Render Enhancing Rim
        if "enhancing" in self.tumor_meshes:
            t = self.tumor_meshes["enhancing"]
            self._render_shaded_mesh(
                frame, t["mesh"], camera_matrix, dist_coeffs,
                base_color=t["color"], alpha=t["alpha"], draw_wire=True
            )
            
        # Render Necrotic Core (deepest aggressive center)
        if "necrotic" in self.tumor_meshes:
            t = self.tumor_meshes["necrotic"]
            self._render_shaded_mesh(
                frame, t["mesh"], camera_matrix, dist_coeffs,
                base_color=t["color"], alpha=t["alpha"], draw_wire=True
            )

        # 2. Render Outer Translucent Holographic Brain Cortex Shell
        if show_cortex and self.cortex_mesh is not None:
            self._render_shaded_mesh(
                frame, self.cortex_mesh, camera_matrix, dist_coeffs,
                base_color=(200, 180, 60), # Holographic Cyan-Teal
                alpha=cortex_opacity,
                draw_wire=True
            )

        return frame
