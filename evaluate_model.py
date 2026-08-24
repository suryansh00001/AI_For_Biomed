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
    parser = argparse.ArgumentParser(description="Evaluate model Dice on held-out subjects")
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

    scores = {'whole_tumor': [], 'tumor_core': [], 'enhancing_tumor': []}

    for subj in held_out:
        subj_dir = os.path.join(data_dir, subj)

        def load(mod):
            files = [f for f in os.listdir(subj_dir) if mod + ".nii" in f]
            return nib.load(os.path.join(subj_dir, files[0])).get_fdata(dtype=np.float32)

        flair, t1, t1ce, t2 = load("flair"), load("t1"), load("t1ce"), load("t2")
        gt = load("seg")

        pred, _ = engine.predict_volume_3d(flair, t1, t1ce, t2)

        pred_regions = extract_subregion_masks(pred)
        gt_regions = extract_subregion_masks(gt)

        subj_scores = {}
        for region in scores.keys():
            d = dice_score(pred_regions[region], gt_regions[region])
            if d is not None:
                scores[region].append(d)
                subj_scores[region] = round(d, 4)
        print(f"  {subj}: {subj_scores}")

    print("\n--- Held-Out Evaluation Summary (mean Dice over evaluated subjects) ---")
    for region, vals in scores.items():
        if vals:
            print(f"  {region:>16}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  (n={len(vals)})")
        else:
            print(f"  {region:>16}: no tumor-bearing cases evaluated")


if __name__ == "__main__":
    main()
