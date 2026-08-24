import os
import random
import time
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from data_pipeline.preprocessor import normalize_intensity, remap_labels_to_continuous
from data_pipeline.dataset_loader import get_brats_data_directory, list_subject_dirs
from models.unet2d import UNet2D
from models.losses import CombinedDiceCELoss, compute_brats_dice_scores

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class BraTSTumorSliceDataset(Dataset):
    """
    Focused dataset targeting rich tumor and brain slices for fast training.
    Only the selected slice windows are kept in memory; volumes are read
    subject-by-subject during indexing and released immediately.
    """
    def __init__(self, data_dir, subject_dirs=None, max_subjects=30, slices_per_subj=12):
        self.data_dir = data_dir
        self.samples = []

        if subject_dirs is None:
            subject_dirs = list_subject_dirs(data_dir)
        subj_dirs = sorted(subject_dirs)[:max_subjects]
        print(f"[Dataset] Extracting high-information tumor slices from {len(subj_dirs)} subjects...")

        for s_dir in subj_dirs:
            seg_files = glob.glob(os.path.join(s_dir, "*seg.nii*"))
            flair_files = glob.glob(os.path.join(s_dir, "*flair.nii*"))
            t1_files = glob.glob(os.path.join(s_dir, "*t1.nii*"))
            t1ce_files = glob.glob(os.path.join(s_dir, "*t1ce.nii*"))
            t2_files = glob.glob(os.path.join(s_dir, "*t2.nii*"))

            # All four modalities + ground truth are required
            if not (seg_files and flair_files and t1_files and t1ce_files and t2_files):
                print(f"[Dataset] Skipping {os.path.basename(s_dir)} (missing modalities)")
                continue

            seg_vol = nib.load(seg_files[0]).get_fdata(dtype=np.float32)
            flair_vol = normalize_intensity(nib.load(flair_files[0]).get_fdata(dtype=np.float32))
            t1_vol = normalize_intensity(nib.load(t1_files[0]).get_fdata(dtype=np.float32))
            t1ce_vol = normalize_intensity(nib.load(t1ce_files[0]).get_fdata(dtype=np.float32))
            t2_vol = normalize_intensity(nib.load(t2_files[0]).get_fdata(dtype=np.float32))

            # Find slices with highest tumor content
            tumor_counts = [(z, (seg_vol[:, :, z] > 0).sum()) for z in range(20, 140)]
            tumor_counts.sort(key=lambda x: x[1], reverse=True)

            selected_z = [z for z, cnt in tumor_counts[:slices_per_subj] if cnt > 50]
            # Also add 2 background/boundary slices (deduplicated)
            selected_z = sorted(set(selected_z) | {35, 125})

            for z in selected_z:
                # Center crop to 192x192 (matches inference tile size)
                flair_s = flair_vol[24:216, 24:216, z]
                t1_s = t1_vol[24:216, 24:216, z]
                t1ce_s = t1ce_vol[24:216, 24:216, z]
                t2_s = t2_vol[24:216, 24:216, z]
                seg_s = remap_labels_to_continuous(seg_vol[24:216, 24:216, z])

                # 4-channel image
                img_4ch = np.stack([flair_s, t1_s, t1ce_s, t2_s], axis=0).astype(np.float32)
                self.samples.append((img_4ch, seg_s))

        print(f"[Dataset] Ready with {len(self.samples)} high-quality training slices.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        return {
            'image': torch.from_numpy(img),
            'mask': torch.from_numpy(mask).long()
        }


def train(checkpoint_dir=None, epochs=8):
    data_dir = get_brats_data_directory()
    checkpoint_dir = checkpoint_dir or os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Subject-level split so validation measures generalization to unseen patients
    subjects = [os.path.basename(d) for d in list_subject_dirs(data_dir)]
    rng = random.Random(42)
    rng.shuffle(subjects)
    subjects = subjects[:25]
    n_val = max(1, int(0.2 * len(subjects)))
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])
    print(f"[Split] {len(train_subjects)} train / {n_val} validation subjects "
          f"(validation: {sorted(val_subjects)})")

    all_dirs = {os.path.basename(d): d for d in list_subject_dirs(data_dir)}
    train_ds = BraTSTumorSliceDataset(
        data_dir=data_dir,
        subject_dirs=[all_dirs[s] for s in sorted(train_subjects)],
        slices_per_subj=10
    )
    val_ds = BraTSTumorSliceDataset(
        data_dir=data_dir,
        subject_dirs=[all_dirs[s] for s in sorted(val_subjects)],
        slices_per_subj=20
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    print(f"Training on {len(train_ds)} slices, validating on {len(val_ds)} slices "
          f"from unseen patients...")

    # Model (base_filters=16 matches shipped checkpoints & inference engine)
    model = UNet2D(in_channels=4, num_classes=4, base_filters=16)

    # Use class weights to penalize tumor misclassification
    class_weights = torch.tensor([0.2, 2.5, 1.5, 2.5], dtype=torch.float32)
    criterion = CombinedDiceCELoss(weight_ce=1.0, weight_dice=1.5, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    best_dice = 0.0

    print("\n--- Training BraTS AI Segmentation Model ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            imgs = batch['image']
            masks = batch['mask']

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image']
                masks = batch['mask']
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                for p, t in zip(preds, masks):
                    d = compute_brats_dice_scores(p, t)
                    if d['dice_mean'] == d['dice_mean']:  # skip NaN (empty-empty regions)
                        dice_scores.append(d['dice_mean'])

        val_loss /= len(val_loader)
        mean_dice = sum(dice_scores) / max(1, len(dice_scores))
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {mean_dice:.4f}")

        if dice_scores and mean_dice >= best_dice:
            best_dice = mean_dice
            save_path = os.path.join(checkpoint_dir, "best_unet2d_brats.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best model saved! (Dice: {best_dice:.4f})")

    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "trained_brats_unet.pth"))
    print("\n[Complete] AI Model training complete and checkpoint saved!")


if __name__ == "__main__":
    train()
