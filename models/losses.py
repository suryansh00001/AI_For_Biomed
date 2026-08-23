import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Multiclass Soft Dice Loss.
    """
    def __init__(self, smooth=1e-5, ignore_background=False):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W) or (B, C, H, W, D)
        targets: (B, H, W) or (B, H, W, D) with class indices [0, C-1]
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        # Move channel dimension to dim 1
        if targets.ndim == 3: # 2D: (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        elif targets.ndim == 4: # 3D: (B, H, W, D) -> (B, H, W, D, C) -> (B, C, H, W, D)
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        start_c = 1 if self.ignore_background else 0
        dice_total = 0.0
        active_classes = 0

        for c in range(start_c, num_classes):
            p = probs[:, c].contiguous().view(-1)
            t = targets_one_hot[:, c].contiguous().view(-1)

            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_total += dice
            active_classes += 1

        return 1.0 - (dice_total / max(1, active_classes))

class CombinedDiceCELoss(nn.Module):
    """
    Composite Loss: Dice Loss + Cross-Entropy Loss.
    Addresses extreme class imbalance between healthy brain and small tumor cores.
    """
    def __init__(self, weight_ce=1.0, weight_dice=1.0, class_weights=None):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss

def compute_brats_dice_scores(pred_mask, true_mask):
    """
    Computes standard BraTS evaluation metrics:
      - WT (Whole Tumor): labels {1, 2, 3}
      - TC (Tumor Core): labels {1, 3}
      - ET (Enhancing Tumor): label {3}
    
    Inputs can be numpy arrays or torch tensors with continuous labels {0,1,2,3}.
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.detach().cpu().numpy()

    def dice_binary(p, t):
        p_bool = p.astype(bool)
        t_bool = t.astype(bool)
        if not np.any(p_bool) and not np.any(t_bool):
            return 1.0 # True negative perfect match
        intersection = np.logical_and(p_bool, t_bool).sum()
        return (2.0 * intersection) / (p_bool.sum() + t_bool.sum() + 1e-6)

    # WT: 1, 2, 3
    pred_wt = np.isin(pred_mask, [1, 2, 3])
    true_wt = np.isin(true_mask, [1, 2, 3])
    dice_wt = dice_binary(pred_wt, true_wt)

    # TC: 1, 3 (Necrotic + Enhancing)
    pred_tc = np.isin(pred_mask, [1, 3])
    true_tc = np.isin(true_mask, [1, 3])
    dice_tc = dice_binary(pred_tc, true_tc)

    # ET: 3 (Enhancing)
    pred_et = (pred_mask == 3)
    true_et = (true_mask == 3)
    dice_et = dice_binary(pred_et, true_et)

    return {
        'dice_wt': float(dice_wt),
        'dice_tc': float(dice_tc),
        'dice_et': float(dice_et),
        'dice_mean': float((dice_wt + dice_tc + dice_et) / 3.0)
    }
