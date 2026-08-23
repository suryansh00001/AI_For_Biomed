import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

class BraTS2020Dataset(Dataset):
    """
    PyTorch Dataset for BraTS 2020 Brain Tumor Segmentation & Classification.
    Modalities per subject:
      - flair: FLAIR image (.nii / .nii.gz)
      - t1: T1-weighted image
      - t1ce: T1-contrast enhanced image
      - t2: T2-weighted image
      - seg: Ground truth segmentation mask (0: background, 1: necrotic/non-enhancing core, 2: edema, 4: enhancing tumor)
    """
    def __init__(self, root_dir, is_training=True, transform=None, slice_mode=False):
        self.root_dir = root_dir
        self.is_training = is_training
        self.transform = transform
        self.slice_mode = slice_mode
        
        # Look for subject folders (e.g. BraTS20_Training_001)
        sub_dirs = []
        if os.path.exists(root_dir):
            for entry in os.scandir(root_dir):
                if entry.is_dir() and "BraTS20_" in entry.name:
                    sub_dirs.append(entry.path)
            if not sub_dirs:
                # Recursively look for folders containing flair
                for root, dirs, files in os.walk(root_dir):
                    if any("_flair.nii" in f for f in files):
                        sub_dirs.append(root)
        
        self.subject_paths = sorted(sub_dirs)
        print(f"Loaded {len(self.subject_paths)} subjects from {root_dir}")

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):
        subj_dir = self.subject_paths[idx]
        subj_id = os.path.basename(subj_dir)

        def load_nifti(pattern):
            files = glob.glob(os.path.join(subj_dir, f"*{pattern}*"))
            if files:
                img = nib.load(files[0]).get_fdata(dtype=np.float32)
                return img
            return None

        flair = load_nifti("flair")
        t1 = load_nifti("t1.") or load_nifti("t1_")
        t1ce = load_nifti("t1ce")
        t2 = load_nifti("t2")
        seg = load_nifti("seg") if self.is_training else None

        # Stack the 4 MRI modalities into a multi-channel volume (4, H, W, D)
        channels = [ch for ch in [flair, t1, t1ce, t2] if ch is not None]
        if channels:
            volume = np.stack(channels, axis=0)
        else:
            volume = np.zeros((4, 240, 240, 155), dtype=np.float32)

        sample = {
            'id': subj_id,
            'image': torch.from_numpy(volume),
        }
        if seg is not None:
            sample['mask'] = torch.from_numpy(seg).long()

        return sample

if __name__ == "__main__":
    print("BraTS 2020 Dataset Loader module ready.")
