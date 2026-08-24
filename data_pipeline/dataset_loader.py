import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from data_pipeline.preprocessor import normalize_intensity, remap_labels_to_continuous

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_brats_data_directory():
    """Locates BraTS 2020 training data directory in cache or local workspace."""
    cache_path = os.path.expanduser('~/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
    if os.path.exists(cache_path):
        return cache_path

    # Check local relative paths
    local_candidates = [
        os.path.join(PROJECT_ROOT, 'data', 'BraTS2020_TrainingData'),
        os.path.join(PROJECT_ROOT, 'BraTS2020_TrainingData'),
        os.path.join(PROJECT_ROOT, 'data'),
    ]
    for cand in local_candidates:
        if os.path.exists(cand):
            return cand
    return cache_path


def list_subject_dirs(data_dir=None):
    """Returns sorted full paths of BraTS 2020 subject folders in data_dir."""
    data_dir = data_dir or get_brats_data_directory()
    if not os.path.exists(data_dir):
        return []
    return [
        os.path.join(data_dir, entry)
        for entry in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, entry)) and "BraTS20_" in entry
    ]

class BraTS3DDataset(Dataset):
    """
    3D Multi-modal Dataset for BraTS 2020.
    Output:
      - image: Tensor (4, H, W, D) representing [FLAIR, T1, T1ce, T2]
      - mask: Tensor (H, W, D) with remapped continuous classes {0, 1, 2, 3}
    """
    def __init__(self, data_dir=None, is_training=True, max_subjects=None):
        self.data_dir = data_dir or get_brats_data_directory()
        self.is_training = is_training
        
        self.subject_dirs = []
        if os.path.exists(self.data_dir):
            for entry in sorted(os.listdir(self.data_dir)):
                full_path = os.path.join(self.data_dir, entry)
                if os.path.isdir(full_path) and "BraTS20_" in entry:
                    self.subject_dirs.append(full_path)
        
        if max_subjects:
            self.subject_dirs = self.subject_dirs[:max_subjects]
            
        print(f"[BraTS3DDataset] Found {len(self.subject_dirs)} subjects in {self.data_dir}")

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subj_dir = self.subject_dirs[idx]
        subj_id = os.path.basename(subj_dir)

        def load_nii(modality):
            pattern = os.path.join(subj_dir, f"*{modality}.nii*")
            files = glob.glob(pattern)
            if files:
                return nib.load(files[0]).get_fdata(dtype=np.float32)
            raise FileNotFoundError(f"Missing {modality} for {subj_id}")

        flair = normalize_intensity(load_nii("flair"))
        t1 = normalize_intensity(load_nii("t1"))
        t1ce = normalize_intensity(load_nii("t1ce"))
        t2 = normalize_intensity(load_nii("t2"))

        # 4-channel 3D volume: (4, H, W, D)
        volume = np.stack([flair, t1, t1ce, t2], axis=0).astype(np.float32)
        sample = {
            'id': subj_id,
            'image': torch.from_numpy(volume)
        }

        if self.is_training:
            seg_raw = load_nii("seg")
            seg_remapped = remap_labels_to_continuous(seg_raw)
            sample['mask'] = torch.from_numpy(seg_remapped).long()

        return sample

class BraTS2DSliceDataset(Dataset):
    """
    2D Multi-modal Slice Dataset for fast training and real-time slice inference.
    Takes 4-channel 2D axial slices (4, H, W).
    """
    def __init__(self, data_dir=None, is_training=True, max_subjects=50, slice_step=2, filter_empty=True, allowed_subjects=None):
        self.data_dir = data_dir or get_brats_data_directory()
        self.is_training = is_training
        self.slice_entries = [] # List of (subj_dir, slice_idx)

        subject_dirs = list_subject_dirs(self.data_dir)

        if allowed_subjects is not None:
            allowed = set(allowed_subjects)
            subject_dirs = [d for d in subject_dirs if os.path.basename(d) in allowed]

        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]

        print(f"[BraTS2DSliceDataset] Indexing slices across {len(subject_dirs)} subjects...")
        
        for s_dir in subject_dirs:
            seg_files = glob.glob(os.path.join(s_dir, "*seg.nii*"))
            if seg_files and is_training:
                # Fast header check or load mask to find slices with brain/tumor
                seg = nib.load(seg_files[0]).get_fdata(dtype=np.float32)
                for z in range(30, 140, slice_step): # Brain is concentrated between slice 30 and 140
                    slice_mask = seg[:, :, z]
                    if filter_empty:
                        # Include slices with tumor or non-zero brain tissue
                        if np.any(slice_mask > 0) or (z % 10 == 0 and np.any(slice_mask == 0)):
                            self.slice_entries.append((s_dir, z))
                    else:
                        self.slice_entries.append((s_dir, z))
            else:
                for z in range(30, 140, slice_step):
                    self.slice_entries.append((s_dir, z))

        print(f"[BraTS2DSliceDataset] Total 2D slices indexed: {len(self.slice_entries)}")

    def __len__(self):
        return len(self.slice_entries)

    def __getitem__(self, idx):
        subj_dir, z = self.slice_entries[idx]
        subj_id = os.path.basename(subj_dir)

        def load_slice(modality):
            files = glob.glob(os.path.join(subj_dir, f"*{modality}.nii*"))
            if files:
                vol = nib.load(files[0]).get_fdata(dtype=np.float32)
                s = vol[:, :, z]
                return normalize_intensity(s)
            raise FileNotFoundError(f"Missing {modality}")

        flair_s = load_slice("flair")
        t1_s = load_slice("t1")
        t1ce_s = load_slice("t1ce")
        t2_s = load_slice("t2")

        # 4-channel 2D image: (4, 240, 240)
        img_2d = np.stack([flair_s, t1_s, t1ce_s, t2_s], axis=0).astype(np.float32)

        sample = {
            'id': subj_id,
            'slice_idx': z,
            'image': torch.from_numpy(img_2d)
        }

        if self.is_training:
            seg_files = glob.glob(os.path.join(subj_dir, "*seg.nii*"))
            if seg_files:
                seg_vol = nib.load(seg_files[0]).get_fdata(dtype=np.float32)
                seg_s = remap_labels_to_continuous(seg_vol[:, :, z])
                sample['mask'] = torch.from_numpy(seg_s).long()

        return sample
