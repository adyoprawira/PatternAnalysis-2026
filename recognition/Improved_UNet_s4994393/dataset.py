from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class HipMRIDataset(Dataset):
    def __init__(self, root, split="train"):
        # Validate dataset split
        if split not in ["train", "validate", "test"]:
            raise ValueError(f"Invalid split: {split}")

        # Locate MRI images and segmentation masks
        self.image_dir = Path(root) / f"keras_slices_{split}"
        self.mask_dir = Path(root) / f"keras_slices_seg_{split}"

        # Find and sort all MRI images
        self.images = sorted(self.image_dir.glob("case_*.nii.gz"))  

        if not self.images:
            raise ValueError(f"No MRI images found in {self.image_dir}")

        # Match each MRI image to its segmentation mask
        self.masks = [
            self.mask_dir / image.name.replace("case_", "seg_", 1)
            for image in self.images
        ]

        # Verify that every MRI has a corresponding mask
        for mask in self.masks:
            if not mask.exists():
                raise FileNotFoundError(f"Missing mask: {mask}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load MRI and corresponding segmentation mask
        image = nib.load(self.images[idx]).get_fdata()
        mask = nib.load(self.masks[idx]).get_fdata()

        # Convert to appropriate data types
        image = image.astype(np.float32)
        mask = np.rint(mask).astype(np.int64)

        # Normalize MRI intensities
        image = (image - image.mean()) / (image.std() + 1e-8)

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask)
    
        return image, mask       