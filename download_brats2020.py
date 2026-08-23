import os
import kagglehub
import nibabel as nib
import numpy as np

def download_and_inspect_brats2020():
    print("Starting download of BraTS 2020 dataset from Kaggle...")
    # Download dataset via kagglehub (uses cache automatically)
    path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
    print(f"\nDataset downloaded successfully!")
    print(f"Location: {path}")

    print("\nExploring dataset structure...")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        # show first 5 files if any

        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
        if level >= 2:
            dirs.clear() # don't recurse too deep

if __name__ == "__main__":
    download_and_inspect_brats2020()
