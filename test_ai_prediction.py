import os
import glob
import nibabel as nib
import numpy as np
from data_pipeline.dataset_loader import get_brats_data_directory
from models.inference_engine import BraTSInferenceEngine
from reconstruction.mesh_generator import BrainMeshReconstructor
from reconstruction.exporter import MeshExporter

def test_inference():
    data_dir = get_brats_data_directory()
    patient_id = "BraTS20_Training_002"
    patient_dir = os.path.join(data_dir, patient_id)
    
    print(f"=== Testing Full AI Inference on Raw MRI: {patient_id} ===")
    
    # 1. Load Raw MRI scans (FLAIR, T1, T1ce, T2) - WITHOUT using ground truth seg
    flair = nib.load(glob.glob(os.path.join(patient_dir, "*flair.nii*"))[0]).get_fdata(dtype=np.float32)
    t1 = nib.load(glob.glob(os.path.join(patient_dir, "*t1.nii*"))[0]).get_fdata(dtype=np.float32)
    t1ce = nib.load(glob.glob(os.path.join(patient_dir, "*t1ce.nii*"))[0]).get_fdata(dtype=np.float32)
    t2 = nib.load(glob.glob(os.path.join(patient_dir, "*t2.nii*"))[0]).get_fdata(dtype=np.float32)
    
    print(f"Input Raw MRI Scans loaded: shape = {flair.shape}")

    # 2. Initialize trained AI Deep Learning model
    engine = BraTSInferenceEngine(checkpoint_path="d:/Brain tumorr/checkpoints/best_unet2d_brats.pth")

    # 3. Predict 3D Tumor Segmentation & Brain Extraction
    pred_seg_3d, brain_mask_3d = engine.predict_volume_3d(flair, t1, t1ce, t2)

    # 4. Volumetric Quantification from AI Prediction
    brain_vol_cm3 = float(brain_mask_3d.sum() * 0.001)
    edema_vol_cm3 = float((pred_seg_3d == 2).sum() * 0.001)
    ncr_vol_cm3 = float((pred_seg_3d == 1).sum() * 0.001)
    et_vol_cm3 = float((pred_seg_3d == 4).sum() * 0.001)
    wt_vol_cm3 = edema_vol_cm3 + ncr_vol_cm3 + et_vol_cm3

    print("\n--- AI Model Prediction Results ---")
    print(f"  • Brain Volume: {brain_vol_cm3:.1f} cm³")
    print(f"  • Whole Tumor (Predicted): {wt_vol_cm3:.1f} cm³")
    print(f"  • Edema (Predicted): {edema_vol_cm3:.1f} cm³")
    print(f"  • Necrotic Core (Predicted): {ncr_vol_cm3:.1f} cm³")
    print(f"  • Enhancing Core (Predicted): {et_vol_cm3:.1f} cm³")
    print(f"  • Tumor Burden: {(wt_vol_cm3 / brain_vol_cm3 * 100):.2f}%")

    # 5. Stage 2: 3D Mesh Reconstruction directly from the AI model's predicted mask
    print("\nReconstructing 3D polygonal surface meshes from AI predictions...")
    reconstructor = BrainMeshReconstructor(voxel_spacing=(1.0, 1.0, 1.0))
    meshes = reconstructor.reconstruct_full_patient_scene(flair, pred_seg_3d, center_at_origin=True)

    exporter = MeshExporter(output_dir="d:/Brain tumorr/exported_3d_models")
    exports = exporter.export_patient_scene(f"{patient_id}_AI_PREDICTED", meshes)
    print("3D Model exported for Unity/WebGL at:", exports.get('composite_glb'))

if __name__ == "__main__":
    test_inference()
