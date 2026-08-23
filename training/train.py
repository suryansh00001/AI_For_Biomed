import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_pipeline.dataset_loader import BraTS2DSliceDataset, get_brats_data_directory
from models.unet2d import UNet2D
from models.losses import CombinedDiceCELoss, compute_brats_dice_scores

def train_brats_model(
    epochs=10,
    batch_size=16,
    lr=1e-3,
    max_subjects=40,
    checkpoint_dir="d:/Brain tumorr/checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    # 1. Dataset & DataLoader
    print(f"[Training] Loading BraTS dataset (indexing up to {max_subjects} subjects)...")
    full_dataset = BraTS2DSliceDataset(max_subjects=max_subjects, slice_step=3, filter_empty=True)
    
    val_size = max(1, int(0.15 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Training] Training slices: {train_size}, Validation slices: {val_size}")

    # 2. Model, Loss, Optimizer
    model = UNet2D(in_channels=4, num_classes=4, base_filters=32).to(device)
    criterion = CombinedDiceCELoss(weight_ce=1.0, weight_dice=1.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    print("\n--- Starting Training Loop ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device) # (B, 4, 240, 240)
            masks = batch['mask'].to(device)   # (B, 240, 240)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        dice_scores_list = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                for p, t in zip(preds, masks):
                    d = compute_brats_dice_scores(p, t)
                    dice_scores_list.append(d['dice_mean'])

        val_loss /= max(1, len(val_loader))
        avg_val_dice = sum(dice_scores_list) / max(1, len(dice_scores_list))
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean Dice: {avg_val_dice:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(avg_val_dice)

        # Save Best Model
        if avg_val_dice >= best_val_dice:
            best_val_dice = avg_val_dice
            ckpt_path = os.path.join(checkpoint_dir, "best_unet2d_brats.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': avg_val_dice,
            }, ckpt_path)
            print(f"  -> Saved best model checkpoint to {ckpt_path} (Dice: {best_val_dice:.4f})")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_unet2d_brats.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining Complete! Final model saved to {final_path}")
    return model, history

if __name__ == "__main__":
    train_brats_model(epochs=5, batch_size=8, max_subjects=20)
