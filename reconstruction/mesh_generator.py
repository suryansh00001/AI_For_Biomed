import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import trimesh
from data_pipeline.preprocessor import extract_brain_mask, extract_subregion_masks

class BrainMeshReconstructor:
    """
    Stage 2: 3D Surface Mesh Reconstruction from 3D MRI voxel segmentations.
    Generates optimized polygonal meshes using Marching Cubes, Laplacian smoothing, and decimation.
    """
    def __init__(self, voxel_spacing=(1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing

    def extract_mesh_from_binary_mask(self, mask_3d, step_size=1, smooth_iterations=10, decimation_ratio=0.5):
        """
        Runs Marching Cubes on a 3D binary volume mask and returns a smoothed Trimesh object.
        """
        if not np.any(mask_3d):
            return None

        # Slight Gaussian smoothing on binary volume to eliminate voxel stepping before marching cubes
        smooth_vol = ndi.gaussian_filter(mask_3d.astype(np.float32), sigma=0.8)
        
        # Marching cubes isosurface extraction at threshold 0.5
        try:
            verts, faces, normals, values = measure.marching_cubes(
                smooth_vol,
                level=0.5,
                spacing=self.voxel_spacing,
                step_size=step_size
            )
        except Exception as e:
            print(f"[Marching Cubes] Warning: {e}")
            return None

        # Create Trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

        # Smooth mesh using Laplacian filter
        if smooth_iterations > 0 and len(mesh.vertices) > 0:
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)
            except Exception:
                pass

        # Decimate / simplify mesh if polygon count is very large
        if decimation_ratio < 1.0 and len(mesh.faces) > 2000:
            target_faces = max(1000, int(len(mesh.faces) * decimation_ratio))
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
            except Exception:
                pass

        return mesh

    def reconstruct_full_patient_scene(self, mri_flair_or_t1, seg_mask, center_at_origin=True):
        """
        Reconstructs all anatomical layers for a patient:
          1. Brain Cortex Envelope (Brain Parenchyma)
          2. Edema Sub-region (ED)
          3. Necrotic & Non-Enhancing Core (NCR)
          4. Enhancing Active Tumor (ET)
        """
        print("[3D Reconstruction] Extracting brain parenchyma surface...")
        brain_mask = extract_brain_mask(mri_flair_or_t1)
        brain_mesh = self.extract_mesh_from_binary_mask(brain_mask, step_size=2, smooth_iterations=12, decimation_ratio=0.3)

        subregions = extract_subregion_masks(seg_mask)
        
        print("[3D Reconstruction] Extracting tumor sub-region surfaces...")
        edema_mesh = self.extract_mesh_from_binary_mask(subregions['edema'], step_size=1, smooth_iterations=8, decimation_ratio=0.6)
        ncr_mesh = self.extract_mesh_from_binary_mask(subregions['necrotic_core'], step_size=1, smooth_iterations=8, decimation_ratio=0.6)
        et_mesh = self.extract_mesh_from_binary_mask(subregions['enhancing_tumor'], step_size=1, smooth_iterations=8, decimation_ratio=0.6)
        
        meshes = {
            'brain_cortex': brain_mesh,
            'tumor_edema': edema_mesh,
            'tumor_necrotic': ncr_mesh,
            'tumor_enhancing': et_mesh,
        }

        # Center all meshes at origin based on brain centroid
        if center_at_origin and brain_mesh is not None:
            centroid = brain_mesh.centroid
            for k, m in meshes.items():
                if m is not None:
                    m.apply_translation(-centroid)

        return meshes

if __name__ == "__main__":
    reconstructor = BrainMeshReconstructor()
    print("BrainMeshReconstructor ready.")
