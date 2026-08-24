import os
import numpy as np
import torch
from data_pipeline.preprocessor import normalize_intensity, extract_brain_mask, restore_original_brats_labels
from models.unet2d import UNet2D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CHECKPOINTS = [
    os.path.join(PROJECT_ROOT, "checkpoints", "best_unet2d_brats.pth"),
    os.path.join(PROJECT_ROOT, "checkpoints", "trained_brats_unet.pth"),
]


class BraTSInferenceEngine:
    """
    Inference Engine: Runs the trained 2D U-Net over 3D MRI volumes using
    overlapping sliding-window tiles with Gaussian-weighted probability blending.

    Works on any input resolution: inputs smaller than the tile size are padded,
    larger inputs are processed with a strided tile grid (no center-crop blind spot).
    """

    TILE_SIZE = 192   # matches the 192x192 crops used at training time
    OVERLAP = 32      # context overlap between adjacent tiles
    NUM_CLASSES = 4

    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet2D(in_channels=4, num_classes=self.NUM_CLASSES, base_filters=16).to(self.device)
        self.is_trained = False

        candidates = [checkpoint_path] + DEFAULT_CHECKPOINTS
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

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare_input(flair_slice, t1_slice, t1ce_slice, t2_slice):
        """Stacks four normalized 2D modalities into a (4, H, W) float32 array."""
        return np.stack([
            normalize_intensity(flair_slice),
            normalize_intensity(t1_slice),
            normalize_intensity(t1ce_slice),
            normalize_intensity(t2_slice),
        ], axis=0).astype(np.float32)

    def _starts(self, dim):
        """Tile start offsets covering `dim` with stride = TILE_SIZE - OVERLAP."""
        tile = self.TILE_SIZE
        stride = tile - self.OVERLAP
        if dim <= tile:
            return [0]
        starts = list(range(0, dim - tile + 1, stride))
        if starts[-1] != dim - tile:
            starts.append(dim - tile)
        return starts

    @staticmethod
    def _gaussian_importance(tile):
        """2D Gaussian importance map (nnU-Net style) so tile centers count more."""
        center = (tile - 1) / 2.0
        sigma = tile / 8.0
        g = np.exp(-0.5 * ((np.arange(tile) - center) / sigma) ** 2)
        imp = np.outer(g, g)
        return (imp / imp.max()).astype(np.float32)

    def _segment_4ch(self, image_4ch, batch_size=8):
        """
        Sliding-window segmentation of a normalized (4, H, W) array.
        Returns an (H, W) int64 label map with continuous classes {0..3}.
        """
        _, h, w = image_4ch.shape
        tile = self.TILE_SIZE

        pad_h, pad_w = max(0, tile - h), max(0, tile - w)
        if pad_h or pad_w:
            image_4ch = np.pad(image_4ch, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

        H, W = image_4ch.shape[1:]
        windows = [(y, x) for y in self._starts(H) for x in self._starts(W)]

        prob = np.zeros((self.NUM_CLASSES, H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)
        importance = self._gaussian_importance(tile)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                chunk = windows[i:i + batch_size]
                tiles = np.stack([image_4ch[:, y:y + tile, x:x + tile] for y, x in chunk])
                logits = self.model(torch.from_numpy(tiles).to(self.device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for (y, x), p in zip(chunk, probs):
                    prob[:, y:y + tile, x:x + tile] += p * importance
                    weight[y:y + tile, x:x + tile] += importance

        pred = np.argmax(prob / np.maximum(weight, 1e-8), axis=0)
        return pred[:h, :w].astype(np.int64)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_slice(self, flair_slice, t1_slice, t1ce_slice, t2_slice):
        """
        Runs AI inference on a single 2D multi-modal slice of arbitrary size.
        Returns: 2D numpy array with BraTS predicted labels {0, 1, 2, 4}.
        """
        image = self._prepare_input(flair_slice, t1_slice, t1ce_slice, t2_slice)
        return restore_original_brats_labels(self._segment_4ch(image))

    def predict_volume_3d(self, flair_vol, t1_vol, t1ce_vol, t2_vol, batch_size=8):
        """
        Runs AI inference across the entire 3D MRI volume.
        Returns:
          - pred_seg_3d: numpy array with predicted tumor labels {0, 1, 2, 4}
          - brain_mask_3d: boolean numpy array of segmented brain parenchyma
        """
        depth = flair_vol.shape[2]
        pred_seg_3d = np.zeros_like(flair_vol, dtype=np.int64)
        brain_mask_3d = extract_brain_mask(flair_vol)

        print(f"[InferenceEngine] Running sliding-window AI segmentation across all {depth} 3D slices...")

        for z in range(depth):
            # Skip slices entirely outside the brain (large speedup, no info lost
            # because predictions are masked to the brain anyway).
            if not brain_mask_3d[:, :, z].any():
                continue
            image = self._prepare_input(
                flair_vol[:, :, z], t1_vol[:, :, z], t1ce_vol[:, :, z], t2_vol[:, :, z]
            )
            pred_seg_3d[:, :, z] = restore_original_brats_labels(
                self._segment_4ch(image, batch_size=batch_size)
            )

        # Enforce that tumor can only exist inside the extracted brain mask
        pred_seg_3d[~brain_mask_3d] = 0

        print(f"[InferenceEngine] Segmentation complete. Predicted tumor voxels: {(pred_seg_3d > 0).sum()}")
        return pred_seg_3d, brain_mask_3d
