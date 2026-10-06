"""
Held-out evaluation of the shipped segmentation checkpoint.

Computes standard BraTS region Dice scores (WT / TC / ET) on subjects that were
NOT part of the seed-42 training subset used by train_model.py, so the numbers
reflect generalization to unseen patients rather than memorized slices.

Usage:
    python evaluate_model.py                  # evaluate on 5 held-out subjects
    python evaluate_model.py --max-subjects 10
"""
import os
import argparse
import random
import numpy as np
import nibabel as nib

from data_pipeline.dataset_loader import get_brats_data_directory, list_subject_dirs
from data_pipeline.preprocessor import extract_subregion_masks
from models.inference_engine import BraTSInferenceEngine

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# train_model.py consumes the first 25 subjects of the seed-42 shuffle.
TRAIN_SUBJECT_COUNT = 25


def calculate_segmentation_metrics(pred_bool, true_bool, brain_mask=None):
    """
    Computes comprehensive clinical segmentation metrics for two boolean masks:
    - F1-Score / Dice Similarity Coefficient
    - Sensitivity (Recall / True Positive Rate)
    - Specificity (True Negative Rate)
    - Voxel-wise Accuracy & Balanced Accuracy
    - Precision (Positive Predictive Value)
    """
    if brain_mask is not None:
        p = pred_bool[brain_mask]
        t = true_bool[brain_mask]
    else:
        p = pred_bool.ravel()
        t = true_bool.ravel()

    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    tn = float(np.logical_and(~p, ~t).sum())

    total = tp + fp + fn + tn

    # If both true and pred are empty, region is absent in this patient
    if (tp + fn) == 0 and (tp + fp) == 0:
        return None

    # Dice / F1-Score
    dice_f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0

    # Sensitivity (Recall / TPR) - critical for zero-miss tumor detection
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Specificity (TNR) - critical for preserving healthy brain cortex
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0

    # Precision (PPV)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # Voxel Accuracy
    accuracy = (tp + tn) / total if total > 0 else 1.0

    # Balanced Accuracy (balances foreground and background)
    balanced_acc = (sensitivity + specificity) / 2.0

    return {
        'dice_f1': dice_f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
    }


def dice_score(pred_bool, true_bool):
    """Dice for two boolean masks. Returns None when both are empty (undefined)."""
    denom = pred_bool.sum() + true_bool.sum()
    if denom == 0:
        return None
    return float(2.0 * np.logical_and(pred_bool, true_bool).sum() / denom)


def get_held_out_subjects(data_dir):
    subjects = [os.path.basename(d) for d in list_subject_dirs(data_dir)]
    rng = random.Random(42)
    rng.shuffle(subjects)
    if len(subjects) <= TRAIN_SUBJECT_COUNT:
        # Small local datasets: hold out the last 20%
        n_val = max(1, int(0.2 * len(subjects)))
        return subjects[-n_val:]
    return subjects[TRAIN_SUBJECT_COUNT:]


def main():
    parser = argparse.ArgumentParser(description="Evaluate model Dice, Sensitivity, Specificity, Accuracy on held-out subjects")
    parser.add_argument("--max-subjects", type=int, default=5,
                        help="Number of held-out subjects to evaluate (CPU inference is slow)")
    args = parser.parse_args()

    data_dir = get_brats_data_directory()
    held_out = get_held_out_subjects(data_dir)[:args.max_subjects]
    if not held_out:
        print("No evaluation subjects found. Download the dataset first.")
        return

    print(f"Evaluating on {len(held_out)} held-out subject(s): {held_out}")
    engine = BraTSInferenceEngine(
        checkpoint_path=os.path.join(PROJECT_ROOT, "checkpoints", "best_unet2d_brats.pth")
    )

    regions = ['whole_tumor', 'tumor_core', 'enhancing_tumor']
    metrics_log = {
        r: {'dice_f1': [], 'sensitivity': [], 'specificity': [], 'accuracy': [], 'balanced_accuracy': [], 'precision': []}
        for r in regions
    }

    for subj in held_out:
        subj_dir = os.path.join(data_dir, subj)

        def load(mod):
            files = [f for f in os.listdir(subj_dir) if mod + ".nii" in f]
            return nib.load(os.path.join(subj_dir, files[0])).get_fdata(dtype=np.float32)

        flair, t1, t1ce, t2 = load("flair"), load("t1"), load("t1ce"), load("t2")
        gt = load("seg")

        pred, brain_mask = engine.predict_volume_3d(flair, t1, t1ce, t2)

        pred_regions = extract_subregion_masks(pred)
        gt_regions = extract_subregion_masks(gt)

        subj_report = {}
        for region in regions:
            m = calculate_segmentation_metrics(pred_regions[region], gt_regions[region], brain_mask=brain_mask > 0)
            if m is not None:
                for k, v in m.items():
                    metrics_log[region][k].append(v)
                subj_report[region] = {
                    'F1/Dice': round(m['dice_f1'], 4),
                    'Sensitivity': round(m['sensitivity'], 4),
                    'Specificity': round(m['specificity'], 4),
                    'Accuracy': f"{m['accuracy'] * 100:.2f}%"
                }
        print(f"  {subj}: {subj_report}")

    print("\n" + "=" * 80)
    print("--- Held-Out Evaluation Summary (Mean +/- Std over evaluated subjects) ---")
    print("=" * 80)
    print(f"{'Region':<16} | {'F1 / Dice':<14} | {'Sensitivity':<14} | {'Specificity':<14} | {'Voxel Acc':<11} | {'Bal Acc':<11}")
    print("-" * 80)
    for region in regions:
        d_vals = metrics_log[region]['dice_f1']
        s_vals = metrics_log[region]['sensitivity']
        sp_vals = metrics_log[region]['specificity']
        acc_vals = metrics_log[region]['accuracy']
        bal_vals = metrics_log[region]['balanced_accuracy']
        if d_vals:
            print(
                f"{region:<16} | "
                f"{np.mean(d_vals):.4f} +/- {np.std(d_vals):.3f} | "
                f"{np.mean(s_vals):.4f} +/- {np.std(s_vals):.3f} | "
                f"{np.mean(sp_vals):.4f} +/- {np.std(sp_vals):.3f} | "
                f"{np.mean(acc_vals)*100:.2f}%     | "
                f"{np.mean(bal_vals)*100:.2f}%"
            )
        else:
            print(f"{region:<16} | no tumor-bearing cases evaluated")
    print("=" * 80)


if __name__ == "__main__":
    main()
