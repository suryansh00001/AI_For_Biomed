import numpy as np
import scipy.ndimage as ndi

def normalize_intensity(volume, mask=None):
    """
    Z-score intensity normalization on non-zero brain voxels.
    Formula: (I - mu) / sigma
    """
    if mask is None:
        mask = volume > 0
    
    if np.any(mask):
        mean = np.mean(volume[mask])
        std = np.std(volume[mask])
        if std > 1e-6:
            normalized = np.zeros_like(volume, dtype=np.float32)
            normalized[mask] = (volume[mask] - mean) / std
            return normalized
    
    # Fallback min-max
    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min > 1e-6:
        return (volume - v_min) / (v_max - v_min)
    return volume.astype(np.float32)

def extract_brain_mask(volume, threshold_quantile=0.10):
    """
    Extract brain parenchyma mask from multimodal MRI (e.g. FLAIR or T1).
    Performs binary thresholding, morphological closing and filling holes.
    """
    non_zero = volume[volume > 0]
    if len(non_zero) == 0:
        return np.zeros_like(volume, dtype=bool)
    
    thresh = np.percentile(non_zero, threshold_quantile * 100)
    binary_mask = volume > thresh
    
    # Morphological operations to clean up brain contour
    struct = ndi.generate_binary_structure(3, 1)
    cleaned_mask = ndi.binary_closing(binary_mask, structure=struct, iterations=2)
    cleaned_mask = ndi.binary_fill_holes(cleaned_mask)
    
    # Keep largest connected component (the brain)
    labeled, num_features = ndi.label(cleaned_mask)
    if num_features > 0:
        sizes = ndi.sum(cleaned_mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        brain_mask = labeled == largest_label
        return brain_mask
    return cleaned_mask

def get_bounding_box(mask, margin=5):
    """
    Find 3D bounding box coordinates around the mask.
    Returns: (min_x, max_x), (min_y, max_y), (min_z, max_z)
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    
    min_x, min_y, min_z = coords.min(axis=0)
    max_x, max_y, max_z = coords.max(axis=0) + 1
    
    # Add margin with boundary clipping
    min_x = max(0, min_x - margin)
    min_y = max(0, min_y - margin)
    min_z = max(0, min_z - margin)
    
    max_x = min(mask.shape[0], max_x + margin)
    max_y = min(mask.shape[1], max_y + margin)
    max_z = min(mask.shape[2], max_z + margin)
    
    return (min_x, max_x), (min_y, max_y), (min_z, max_z)

def crop_to_bbox(volume, bbox):
    """Crop 3D volume or 4D multi-channel volume to bounding box."""
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = bbox
    if volume.ndim == 3:
        return volume[min_x:max_x, min_y:max_y, min_z:max_z]
    elif volume.ndim == 4:
        return volume[:, min_x:max_x, min_y:max_y, min_z:max_z]
    return volume

def remap_labels_to_continuous(mask):
    """
    BraTS 2020 labels:
      0: Background
      1: Necrotic & Non-Enhancing Core (NCR/NET)
      2: Edema (ED)
      4: Enhancing Tumor (ET)
    Remaps:
      0 -> 0 (Background)
      1 -> 1 (NCR/NET)
      2 -> 2 (ED)
      4 -> 3 (ET)
    """
    remapped = np.zeros_like(mask, dtype=np.int64)
    remapped[mask == 1] = 1
    remapped[mask == 2] = 2
    remapped[mask == 4] = 3
    return remapped

def restore_original_brats_labels(mask):
    """Restores continuous 0, 1, 2, 3 back to BraTS labels 0, 1, 2, 4."""
    original = np.zeros_like(mask, dtype=np.int64)
    original[mask == 1] = 1
    original[mask == 2] = 2
    original[mask == 3] = 4
    return original

def extract_subregion_masks(label_mask):
    """
    Extracts the standard clinical sub-regions:
    - Whole Tumor (WT): labels 1, 2, 4 (or continuous 1, 2, 3)
    - Tumor Core (TC): labels 1, 4 (or continuous 1, 3)
    - Enhancing Tumor (ET): label 4 (or continuous 3)
    """
    is_continuous = np.max(label_mask) <= 3
    if is_continuous:
        wt = (label_mask == 1) | (label_mask == 2) | (label_mask == 3)
        tc = (label_mask == 1) | (label_mask == 3)
        et = (label_mask == 3)
        ed = (label_mask == 2)
        ncr = (label_mask == 1)
    else:
        wt = (label_mask == 1) | (label_mask == 2) | (label_mask == 4)
        tc = (label_mask == 1) | (label_mask == 4)
        et = (label_mask == 4)
        ed = (label_mask == 2)
        ncr = (label_mask == 1)
        
    return {
        'whole_tumor': wt,
        'tumor_core': tc,
        'enhancing_tumor': et,
        'edema': ed,
        'necrotic_core': ncr
    }
