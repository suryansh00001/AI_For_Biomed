import os
import time
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_pipeline.preprocessor import normalize_intensity, remap_labels_to_continuous
from data_pipeline.dataset_loader import get_brats_data_directory
from models.unet2d import UNet2D
from models.losses import CombinedDiceCELoss, compute_brats_dice_scores

class BraTSTumorSliceDataset(Dataset):
    """
    Focused dataset targeting rich tumor and brain slices for fast, accurate training.
    """
    def __init__(self, data_dir, max_subjects=30, slices_per_subj=12):
        self.data_dir = data_dir
        self.samples = []
        
        subj_dirs = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir)) if "BraTS20_" in d][:max_subjects]
        print(f"[Dataset] Extracting high-information tumor slices from {len(subj_dirs)} subjects...")

        for s_dir in subj_dirs:
            seg_files = glob.glob(os.path.join(s_dir, "*seg.nii*"))
            flair_files = glob.glob(os.path.join(s_dir, "*flair.nii*"))
            t1_files = glob.glob(os.path.join(s_dir, "*t1.nii*"))
            t1ce_files = glob.glob(os.path.join(s_dir, "*t1ce.nii*"))
            t2_files = glob.glob(os.path.join(s_dir, "*t2.nii*"))

            if not (seg_files and flair_files and t1ce_files):
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
            # Also add 2 background/boundary slices
            selected_z.extend([35, 125])

            for z in selected_z:
                # Downsample 240x240 to 160x160 for fast CPU training & high resolution
                # Or keep 128x128 center crop
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

def train():
    data_dir = get_brats_data_directory()
    checkpoint_dir = "d:/Brain tumorr/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = BraTSTumorSliceDataset(data_dir=data_dir, max_subjects=25, slices_per_subj=10)
    
    # Split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    print(f"Training on {train_size} slices, Validating on {val_size} slices...")

    # Model
    model = UNet2D(in_channels=4, num_classes=4, base_filters=16) # Fast & effective
    
    # Use class weights to penalize tumor misclassification
    class_weights = torch.tensor([0.2, 2.5, 1.5, 2.5], dtype=torch.float32)
    criterion = CombinedDiceCELoss(weight_ce=1.0, weight_dice=1.5, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-4)

    epochs = 8
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
                    dice_scores.append(d['dice_mean'])

        val_loss /= len(val_loader)
        mean_dice = sum(dice_scores) / max(1, len(dice_scores))
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {mean_dice:.4f}")

        if mean_dice >= best_dice:
            best_dice = mean_dice
            save_path = os.path.join(checkpoint_dir, "best_unet2d_brats.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best model saved! (Dice: {best_dice:.4f})")

    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "trained_brats_unet.pth"))
    print("\n[Complete] AI Model training complete and checkpoint saved!")

if __name__ == "__main__":
    train()
