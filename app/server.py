import os
import sys
import glob
import io
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import nibabel as nib
from PIL import Image
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil

from data_pipeline.dataset_loader import get_brats_data_directory
from data_pipeline.preprocessor import normalize_intensity, extract_subregion_masks, remap_labels_to_continuous, restore_original_brats_labels
from reconstruction.mesh_generator import BrainMeshReconstructor
from reconstruction.exporter import MeshExporter
from models.unet2d import UNet2D
from models.inference_engine import BraTSInferenceEngine

app = FastAPI(title="BraTS 2020 Medical AI & 3D Visualizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = get_brats_data_directory()
EXPORT_DIR = "d:/Brain tumorr/exported_3d_models"
UPLOADS_DIR = "d:/Brain tumorr/user_uploads"
STATIC_DIR = "d:/Brain tumorr/app/static"
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

reconstructor = BrainMeshReconstructor()
exporter = MeshExporter(output_dir=EXPORT_DIR)
inference_engine = BraTSInferenceEngine()

# Cache for AI predicted 3D volumes
ai_prediction_cache = {}

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/patients")
def list_patients():
    """Lists all available BraTS patient cases."""
    if not os.path.exists(DATA_DIR):
        return {"patients": []}
    
    patients = []
    for entry in sorted(os.listdir(DATA_DIR)):
        if "BraTS20_" in entry:
            patients.append(entry)
    return {"patients": patients, "count": len(patients)}

def get_patient_files(patient_id):
    patient_dir = os.path.join(DATA_DIR, patient_id)
    if not os.path.exists(patient_dir):
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    def find_file(suffix):
        files = glob.glob(os.path.join(patient_dir, f"*{suffix}.nii*"))
        return files[0] if files else None

    return {
        'flair': find_file("flair"),
        't1': find_file("t1"),
        't1ce': find_file("t1ce"),
        't2': find_file("t2"),
        'seg': find_file("seg")
    }

@app.get("/api/slice")
def get_slice(
    patient_id: str,
    plane: str = Query("axial", pattern="^(axial|coronal|sagittal)$"),
    slice_idx: int = 75,
    modality: str = Query("flair", pattern="^(flair|t1|t1ce|t2)$"),
    mask_source: str = Query("model", pattern="^(model|ground_truth|diff)$"),
    show_tumor: bool = True,
    show_edema: bool = True,
    show_necrotic: bool = True,
    show_enhancing: bool = True,
    alpha: float = 0.55
):
    """
    Renders a 2D MRI slice (PNG) with AI model predicted tumor segmentation, ground-truth mask,
    or a high-contrast Discrepancy Map (AI vs Ground Truth comparison).
    """
    files = get_patient_files(patient_id)
    mod_file = files.get(modality)
    if not mod_file:
        raise HTTPException(status_code=404, detail=f"Modality {modality} not found")

    mri_vol = nib.load(mod_file).get_fdata(dtype=np.float32)
    seg_vol = None
    gt_vol = None
    model_vol = None

    if show_tumor:
        if mask_source in ["model", "diff"]:
            if patient_id not in ai_prediction_cache:
                flair_vol = nib.load(files['flair']).get_fdata(dtype=np.float32)
                t1_vol = nib.load(files['t1']).get_fdata(dtype=np.float32) if files.get('t1') else flair_vol
                t1ce_vol = nib.load(files['t1ce']).get_fdata(dtype=np.float32) if files.get('t1ce') else flair_vol
                t2_vol = nib.load(files['t2']).get_fdata(dtype=np.float32) if files.get('t2') else flair_vol
                pred_mask_3d, _ = inference_engine.predict_volume_3d(flair_vol, t1_vol, t1ce_vol, t2_vol)
                ai_prediction_cache[patient_id] = pred_mask_3d

            model_vol = ai_prediction_cache[patient_id]
            seg_vol = model_vol

        if mask_source in ["ground_truth", "diff"]:
            if files.get('seg'):
                gt_vol = nib.load(files['seg']).get_fdata(dtype=np.float32)
                if mask_source == "ground_truth":
                    seg_vol = gt_vol

    # Slice extraction by anatomical plane
    if plane == "axial": # Slice along Z (0 to 154)
        z = np.clip(slice_idx, 0, mri_vol.shape[2] - 1)
        mri_slice = mri_vol[:, :, z]
        seg_slice = seg_vol[:, :, z] if seg_vol is not None else None
        gt_slice = gt_vol[:, :, z] if gt_vol is not None else None
        model_slice = model_vol[:, :, z] if model_vol is not None else None
    elif plane == "coronal": # Slice along Y (0 to 239)
        y = np.clip(slice_idx, 0, mri_vol.shape[1] - 1)
        mri_slice = mri_vol[:, y, :]
        seg_slice = seg_vol[:, y, :] if seg_vol is not None else None
        gt_slice = gt_vol[:, y, :] if gt_vol is not None else None
        model_slice = model_vol[:, y, :] if model_vol is not None else None
    else: # sagittal: Slice along X (0 to 239)
        x = np.clip(slice_idx, 0, mri_vol.shape[0] - 1)
        mri_slice = mri_vol[x, :, :]
        seg_slice = seg_vol[x, :, :] if seg_vol is not None else None
        gt_slice = gt_vol[x, :, :] if gt_vol is not None else None
        model_slice = model_vol[x, :, :] if model_vol is not None else None

    # Rotate 90 degrees for correct medical orientation
    mri_slice = np.rot90(mri_slice)
    if seg_slice is not None:
        seg_slice = np.rot90(seg_slice)
    if gt_slice is not None:
        gt_slice = np.rot90(gt_slice)
    if model_slice is not None:
        model_slice = np.rot90(model_slice)

    # Normalize MRI to 0-255 grayscale
    m_min, m_max = mri_slice.min(), mri_slice.max()
    if m_max - m_min > 1e-6:
        mri_norm = np.clip((mri_slice - m_min) / (m_max - m_min) * 255.0, 0, 255).astype(np.uint8)
    else:
        mri_norm = np.zeros_like(mri_slice, dtype=np.uint8)

    # Create RGB Image
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)

    # Apply Color Overlays if tumor present
    if show_tumor:
        overlay = rgb.astype(np.float32)
        
        if mask_source == "diff" and gt_slice is not None and model_slice is not None:
            # High-Contrast Discrepancy / Error Map
            # True Positive (Overlap / Agreement): Green
            tp_mask = (model_slice > 0) & (gt_slice > 0)
            overlay[tp_mask] = (1.0 - alpha) * overlay[tp_mask] + alpha * np.array([16, 185, 129])
            
            # False Positive (AI Over-segmentation): Electric Cyan
            fp_mask = (model_slice > 0) & (gt_slice == 0)
            overlay[fp_mask] = (1.0 - alpha) * overlay[fp_mask] + alpha * np.array([56, 189, 248])
            
            # False Negative (AI Under-segmentation / Ground Truth Missed): Magenta
            fn_mask = (gt_slice > 0) & (model_slice == 0)
            overlay[fn_mask] = (1.0 - alpha) * overlay[fn_mask] + alpha * np.array([244, 63, 94])
            
            rgb = np.clip(overlay, 0, 255).astype(np.uint8)
        elif seg_slice is not None:
            # Standard Compartment Color Coding
            if show_edema:
                edema_mask = (seg_slice == 2)
                overlay[edema_mask] = (1.0 - alpha) * overlay[edema_mask] + alpha * np.array([16, 185, 129])
                
            if show_necrotic:
                ncr_mask = (seg_slice == 1)
                overlay[ncr_mask] = (1.0 - alpha) * overlay[ncr_mask] + alpha * np.array([244, 63, 94])
                
            if show_enhancing:
                et_mask = (seg_slice == 4) | (seg_slice == 3)
                overlay[et_mask] = (1.0 - alpha) * overlay[et_mask] + alpha * np.array([245, 158, 11])
                
            rgb = np.clip(overlay, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/api/analytics/{patient_id}")
def get_patient_analytics(patient_id: str, source: str = Query("model")):
    """Calculates volumetric analysis and comparative Dice overlap statistics."""
    files = get_patient_files(patient_id)
    flair_vol = nib.load(files['flair']).get_fdata(dtype=np.float32)
    
    # 1 voxel = 1 mm^3 = 0.001 cm^3
    brain_mask = flair_vol > (0.10 * flair_vol.max())
    brain_volume_cm3 = float(brain_mask.sum() * 0.001)

    # Always ensure model prediction is available for comparison
    if patient_id not in ai_prediction_cache:
        t1_vol = nib.load(files['t1']).get_fdata(dtype=np.float32) if files.get('t1') else flair_vol
        t1ce_vol = nib.load(files['t1ce']).get_fdata(dtype=np.float32) if files.get('t1ce') else flair_vol
        t2_vol = nib.load(files['t2']).get_fdata(dtype=np.float32) if files.get('t2') else flair_vol
        pred_seg_3d, _ = inference_engine.predict_volume_3d(flair_vol, t1_vol, t1ce_vol, t2_vol)
        ai_prediction_cache[patient_id] = pred_seg_3d
    
    pred_vol = ai_prediction_cache[patient_id]
    gt_vol = nib.load(files['seg']).get_fdata(dtype=np.float32) if files.get('seg') else None

    # Calculate Dice coefficient if Ground Truth is present
    dice_score = 0.0
    mismatch_cm3 = 0.0
    if gt_vol is not None:
        gt_wt = (gt_vol > 0)
        pred_wt = (pred_vol > 0)
        intersection = (gt_wt & pred_wt).sum()
        total_wt = gt_wt.sum() + pred_wt.sum()
        if total_wt > 0:
            dice_score = round(float((2.0 * intersection) / total_wt * 100.0), 1)
        mismatch_cm3 = round(float((gt_wt ^ pred_wt).sum() * 0.001), 2)

    seg_vol = gt_vol if source == "ground_truth" and gt_vol is not None else pred_vol

    ncr_voxels = float((seg_vol == 1).sum())
    ed_voxels = float((seg_vol == 2).sum())
    et_voxels = float((seg_vol == 4).sum() + (seg_vol == 3).sum())
    wt_voxels = ncr_voxels + ed_voxels + et_voxels
    wt_cm3 = round(wt_voxels * 0.001, 2)

    tumor_stats = {
        'patient_id': patient_id,
        'source': source,
        'brain_volume_cm3': round(brain_volume_cm3, 2),
        'whole_tumor_cm3': wt_cm3,
        'edema_cm3': round(ed_voxels * 0.001, 2),
        'necrotic_cm3': round(ncr_voxels * 0.001, 2),
        'enhancing_cm3': round(et_voxels * 0.001, 2),
        'tumor_burden_percent': round((wt_cm3 / max(1e-3, brain_volume_cm3)) * 100.0, 2),
        'dice_score_percent': dice_score,
        'mismatch_cm3': mismatch_cm3
    }

    return tumor_stats

@app.post("/api/reconstruct/{patient_id}")
def trigger_3d_reconstruction(patient_id: str, source: str = Query("model")):
    """Triggers Marching Cubes 3D surface mesh generation for AI Model or Ground Truth."""
    files = get_patient_files(patient_id)
    flair_vol = nib.load(files['flair']).get_fdata(dtype=np.float32)
    
    suffix = "_model" if source == "model" else "_gt"

    if source == "model":
        if patient_id in ai_prediction_cache:
            seg_vol = ai_prediction_cache[patient_id]
        else:
            t1_vol = nib.load(files['t1']).get_fdata(dtype=np.float32) if files.get('t1') else flair_vol
            t1ce_vol = nib.load(files['t1ce']).get_fdata(dtype=np.float32) if files.get('t1ce') else flair_vol
            t2_vol = nib.load(files['t2']).get_fdata(dtype=np.float32) if files.get('t2') else flair_vol
            pred_seg_3d, _ = inference_engine.predict_volume_3d(flair_vol, t1_vol, t1ce_vol, t2_vol)
            ai_prediction_cache[patient_id] = pred_seg_3d
            seg_vol = pred_seg_3d
    else:
        seg_vol = nib.load(files['seg']).get_fdata(dtype=np.float32) if files.get('seg') else np.zeros_like(flair_vol)

    meshes = reconstructor.reconstruct_full_patient_scene(flair_vol, seg_vol, center_at_origin=True)
    exported = exporter.export_patient_scene(patient_id, meshes, suffix=suffix)
    return {"status": "success", "patient_id": patient_id, "source": source, "exports": exported}

@app.get("/api/model3d/{patient_id}/glb")
def get_3d_model_glb(patient_id: str, source: str = Query("model")):
    """Serves the composite GLB binary 3D model (AI Model or Ground Truth) for Three.js."""
    suffix = "_model" if source == "model" else "_gt"
    patient_dir = os.path.join(EXPORT_DIR, patient_id)
    glb_path = os.path.join(patient_dir, f"{patient_id}{suffix}_composite.glb")
    
    if not os.path.exists(glb_path):
        # Auto-reconstruct if not yet cached
        trigger_3d_reconstruction(patient_id, source=source)
    
    if not os.path.exists(glb_path):
        # Fallback to general composite if available
        fallback_path = os.path.join(patient_dir, f"{patient_id}_composite.glb")
        if os.path.exists(fallback_path):
            glb_path = fallback_path

    if os.path.exists(glb_path):
        return FileResponse(glb_path, media_type="model/gltf-binary", filename=f"{patient_id}_{source}_3d.glb")
    raise HTTPException(status_code=404, detail=f"3D Model for {source} not found")

@app.get("/api/model3d/{patient_id}/download/{format_type}")
def download_model_archive(patient_id: str, format_type: str = "obj", source: str = Query("model")):
    """Downloads individual OBJ / STL files or full bundle for AI Model or Ground Truth."""
    patient_dir = os.path.join(EXPORT_DIR, patient_id)
    suffix = "_model" if source == "model" else "_gt"
    glb_path = os.path.join(patient_dir, f"{patient_id}{suffix}_composite.glb")
    
    if not os.path.exists(glb_path):
        trigger_3d_reconstruction(patient_id, source=source)

    if format_type == "glb":
        if os.path.exists(glb_path):
            return FileResponse(glb_path, filename=f"{patient_id}_{source}_model.glb")
        fallback_path = os.path.join(patient_dir, f"{patient_id}_composite.glb")
        if os.path.exists(fallback_path):
            return FileResponse(fallback_path, filename=f"{patient_id}_model.glb")
    elif format_type in ["brain_obj", "tumor_obj"]:
        prefix = "brain_cortex" if format_type == "brain_obj" else "tumor_enhancing"
        obj_name = f"{prefix}{suffix}.obj"
        obj_path = os.path.join(patient_dir, obj_name)
        if not os.path.exists(obj_path):
            obj_path = os.path.join(patient_dir, f"{prefix}.obj")
        if os.path.exists(obj_path):
            return FileResponse(obj_path, filename=f"{prefix}_{source}.obj")
    
    # Return manifest
    manifest_name = f"manifest{suffix}.json"
    manifest_path = os.path.join(patient_dir, manifest_name)
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(patient_dir, "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, filename="manifest.json")
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/run-ai-pipeline")
def run_ai_pipeline(patient_id: str = Form(...)):
    """
    Executes the full end-to-end AI pipeline for a given patient:
    1. Loads raw MRI scans (FLAIR, T1, T1ce, T2)
    2. Runs neural network inference for brain & tumor segmentation
    3. Reconstructs 3D polygonal surface meshes
    4. Computes volumetric metrics & returns ready-to-render artifacts
    """
    files = get_patient_files(patient_id)
    flair_vol = nib.load(files['flair']).get_fdata(dtype=np.float32)
    t1_vol = nib.load(files['t1']).get_fdata(dtype=np.float32)
    t1ce_vol = nib.load(files['t1ce']).get_fdata(dtype=np.float32)
    t2_vol = nib.load(files['t2']).get_fdata(dtype=np.float32)

    # 1. AI Inference
    pred_seg_3d, brain_mask_3d = inference_engine.predict_volume_3d(flair_vol, t1_vol, t1ce_vol, t2_vol)
    ai_prediction_cache[patient_id] = pred_seg_3d

    # 2. 3D Mesh Reconstruction from AI Predictions
    meshes = reconstructor.reconstruct_full_patient_scene(flair_vol, pred_seg_3d, center_at_origin=True)
    exported = exporter.export_patient_scene(patient_id, meshes)

    # 3. Analytics
    brain_vol_cm3 = float(brain_mask_3d.sum() * 0.001)
    ed_vol_cm3 = float((pred_seg_3d == 2).sum() * 0.001)
    ncr_vol_cm3 = float((pred_seg_3d == 1).sum() * 0.001)
    et_vol_cm3 = float((pred_seg_3d == 4).sum() * 0.001)
    wt_vol_cm3 = ed_vol_cm3 + ncr_vol_cm3 + et_vol_cm3

    analytics = {
        'brain_volume_cm3': round(brain_vol_cm3, 2),
        'whole_tumor_cm3': round(wt_vol_cm3, 2),
        'edema_cm3': round(ed_vol_cm3, 2),
        'necrotic_cm3': round(ncr_vol_cm3, 2),
        'enhancing_cm3': round(et_vol_cm3, 2),
        'tumor_burden_percent': round((wt_vol_cm3 / max(1e-3, brain_vol_cm3)) * 100.0, 2)
    }

    return {
        "status": "success",
        "patient_id": patient_id,
        "analytics": analytics,
        "exports": exported
    }

@app.post("/api/upload-mri")
async def upload_custom_mri(
    patient_name: str = Form(...),
    flair_file: UploadFile = File(...),
    t1_file: UploadFile = File(None),
    t1ce_file: UploadFile = File(None),
    t2_file: UploadFile = File(None)
):
    """
    Accepts user-uploaded MRI scans, runs AI prediction, and generates 3D meshes.
    """
    clean_name = "".join(c for c in patient_name if c.isalnum() or c in ('_', '-')).strip() or "Custom_Patient"
    patient_dir = os.path.join(DATA_DIR, clean_name)
    os.makedirs(patient_dir, exist_ok=True)

    # Save uploaded files
    flair_path = os.path.join(patient_dir, f"{clean_name}_flair.nii")
    with open(flair_path, "wb") as f:
        shutil.copyfileobj(flair_file.file, f)

    if t1_file:
        t1_path = os.path.join(patient_dir, f"{clean_name}_t1.nii")
        with open(t1_path, "wb") as f:
            shutil.copyfileobj(t1_file.file, f)
    else:
        shutil.copyfile(flair_path, os.path.join(patient_dir, f"{clean_name}_t1.nii"))

    if t1ce_file:
        t1ce_path = os.path.join(patient_dir, f"{clean_name}_t1ce.nii")
        with open(t1ce_path, "wb") as f:
            shutil.copyfileobj(t1ce_file.file, f)
    else:
        shutil.copyfile(flair_path, os.path.join(patient_dir, f"{clean_name}_t1ce.nii"))

    if t2_file:
        t2_path = os.path.join(patient_dir, f"{clean_name}_t2.nii")
        with open(t2_path, "wb") as f:
            shutil.copyfileobj(t2_file.file, f)
    else:
        shutil.copyfile(flair_path, os.path.join(patient_dir, f"{clean_name}_t2.nii"))

    # Execute Pipeline
    return run_ai_pipeline(clean_name)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
