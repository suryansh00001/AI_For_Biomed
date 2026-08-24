import os
import glob
import nibabel as nib
import numpy as np
from data_pipeline.dataset_loader import get_brats_data_directory
from reconstruction.mesh_generator import BrainMeshReconstructor
from reconstruction.exporter import MeshExporter

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def run_test():
    data_dir = get_brats_data_directory()
    patient_dir = os.path.join(data_dir, "BraTS20_Training_001")
    
    if not os.path.exists(patient_dir):
        print(f"Directory not found: {patient_dir}")
        # Search any available subject
        subs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if "BraTS20_" in d]
        if not subs:
            print("No subjects found.")
            return
        patient_dir = subs[0]
        
    subj_id = os.path.basename(patient_dir)
    print(f"Processing 3D Reconstruction for: {subj_id}...")

    # Load FLAIR and Ground Truth Seg
    flair_file = glob.glob(os.path.join(patient_dir, "*flair.nii*"))[0]
    seg_file = glob.glob(os.path.join(patient_dir, "*seg.nii*"))[0]

    flair_vol = nib.load(flair_file).get_fdata(dtype=np.float32)
    seg_vol = nib.load(seg_file).get_fdata(dtype=np.float32)

    print(f"Loaded FLAIR: {flair_vol.shape}, Seg: {seg_vol.shape}")
    print(f"Tumor voxels count: {(seg_vol > 0).sum()}")

    # Stage 2: 3D Surface Reconstruction
    reconstructor = BrainMeshReconstructor(voxel_spacing=(1.0, 1.0, 1.0))
    meshes = reconstructor.reconstruct_full_patient_scene(flair_vol, seg_vol, center_at_origin=True)

    # Export to .OBJ, .STL, and .GLB
    exporter = MeshExporter(output_dir=os.path.join(PROJECT_ROOT, "exported_3d_models"))
    exported = exporter.export_patient_scene(subj_id, meshes)

    print("\n--- 3D Reconstruction & Export Summary ---")
    for k, v in exported.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run_test()
