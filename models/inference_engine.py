import os
import glob
import numpy as np
import nibabel as nib
import torch
from data_pipeline.preprocessor import normalize_intensity, extract_brain_mask, restore_original_brats_labels
from models.unet2d import UNet2D

class BraTSInferenceEngine:
    """
    Inference Engine: Runs the trained AI deep neural network model on raw 3D MRI scans
    to produce real-time brain extraction and multi-compartment tumor segmentation.
    """
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet2D(in_channels=4, num_classes=4, base_filters=16).to(self.device)
        self.is_trained = False
        
        # Look for checkpoints
        candidates = [
            checkpoint_path,
            "d:/Brain tumorr/checkpoints/best_unet2d_brats.pth",
            "d:/Brain tumorr/checkpoints/trained_brats_unet.pth"
        ]
        
        for cand in candidates:
            if cand and os.path.exists(cand):
                try:
                    state = torch.load(cand, map_location=self.device)
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        self.model.load_state_dict(state['model_state_dict'])
                    else:
                        self.model.load_state_dict(state)
                    self.model.eval()
                    self.is_trained = True
                    print(f"[InferenceEngine] Successfully loaded AI model from {cand}")
                    break
                except Exception as e:
                    print(f"[InferenceEngine] Checkpoint load error ({cand}): {e}")

        if not self.is_trained:
            print("[InferenceEngine] Warning: Model initialized with random weights (no checkpoint loaded).")

    def predict_slice(self, flair_slice, t1_slice, t1ce_slice, t2_slice):
        """
        Runs AI inference on a single 2D multi-modal slice.
        Inputs: 2D numpy arrays of shape (240, 240)
        Returns: 2D numpy array with BraTS predicted labels {0, 1, 2, 4}
        """
        # Crop center 192x192 (or pad to multiple of 16)
        h, w = flair_slice.shape
        flair_n = normalize_intensity(flair_slice)
        t1_n = normalize_intensity(t1_slice)
        t1ce_n = normalize_intensity(t1ce_slice)
        t2_n = normalize_intensity(t2_slice)

        # Center crop from 240x240 to 192x192
        pad_y = (240 - 192) // 2
        pad_x = (240 - 192) // 2
        
        cropped_4ch = np.stack([
            flair_n[pad_y:pad_y+192, pad_x:pad_x+192],
            t1_n[pad_y:pad_y+192, pad_x:pad_x+192],
            t1ce_n[pad_y:pad_y+192, pad_x:pad_x+192],
            t2_n[pad_y:pad_y+192, pad_x:pad_x+192]
        ], axis=0).astype(np.float32)

        tensor_in = torch.from_numpy(cropped_4ch).unsqueeze(0).to(self.device) # (1, 4, 192, 192)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_in)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy() # (192, 192) with {0, 1, 2, 3}

        # Place back in full 240x240 frame
        full_pred = np.zeros((240, 240), dtype=np.int64)
        full_pred[pad_y:pad_y+192, pad_x:pad_x+192] = preds

        # Restore labels to BraTS format: 0, 1, 2, 4
        return restore_original_brats_labels(full_pred)

    def predict_volume_3d(self, flair_vol, t1_vol, t1ce_vol, t2_vol, batch_size=16):
        """
        Runs AI inference across the entire 3D MRI volume (240, 240, 155).
        Returns:
          - pred_seg_3d: (240, 240, 155) numpy array with predicted tumor labels {0, 1, 2, 4}
          - brain_mask_3d: (240, 240, 155) boolean numpy array of segmented brain parenchyma
        """
        depth = flair_vol.shape[2]
        pred_seg_3d = np.zeros_like(flair_vol, dtype=np.int64)
        brain_mask_3d = extract_brain_mask(flair_vol)

        print(f"[InferenceEngine] Running AI Model segmentation across all {depth} 3D slices...")

        # Batch 2D slices
        slice_batches = []
        indices_batches = []
        cur_batch = []
        cur_idx = []

        pad_y = (240 - 192) // 2
        pad_x = (240 - 192) // 2

        for z in range(depth):
            flair_n = normalize_intensity(flair_vol[:, :, z])
            t1_n = normalize_intensity(t1_vol[:, :, z])
            t1ce_n = normalize_intensity(t1ce_vol[:, :, z])
            t2_n = normalize_intensity(t2_vol[:, :, z])

            cropped = np.stack([
                flair_n[pad_y:pad_y+192, pad_x:pad_x+192],
                t1_n[pad_y:pad_y+192, pad_x:pad_x+192],
                t1ce_n[pad_y:pad_y+192, pad_x:pad_x+192],
                t2_n[pad_y:pad_y+192, pad_x:pad_x+192]
            ], axis=0).astype(np.float32)

            cur_batch.append(cropped)
            cur_idx.append(z)

            if len(cur_batch) >= batch_size or z == depth - 1:
                slice_batches.append(np.stack(cur_batch, axis=0))
                indices_batches.append(cur_idx)
                cur_batch = []
                cur_idx = []

        self.model.eval()
        with torch.no_grad():
            for batch_data, idx_list in zip(slice_batches, indices_batches):
                tensor_in = torch.from_numpy(batch_data).to(self.device)
                logits = self.model(tensor_in)
                preds = torch.argmax(logits, dim=1).cpu().numpy() # (B, 192, 192)

                for i, z in enumerate(idx_list):
                    pred_s = preds[i]
                    # Map into 3D volume
                    pred_seg_3d[pad_y:pad_y+192, pad_x:pad_x+192, z] = restore_original_brats_labels(pred_s)

        # Enforce that tumor can only exist inside the extracted brain mask
        pred_seg_3d[~brain_mask_3d] = 0

        print(f"[InferenceEngine] Segmentation complete. Predicted tumor voxels: {(pred_seg_3d > 0).sum()}")
        return pred_seg_3d, brain_mask_3d
