import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BraTSNpy2D(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        mode: str = "binary",
        axis: int = 0,
        only_lesion_slices: bool = True,
        lesion_min_pixels: int = 10,
        max_empty_per_volume: int = 0, 
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mode = mode
        self.axis = axis
        self.only_lesion_slices = only_lesion_slices
        self.lesion_min_pixels = lesion_min_pixels
        self.max_empty_per_volume = max_empty_per_volume

        self.img_files = sorted([p for p in self.images_dir.iterdir() if p.suffix == ".npy" and "image" in p.name])
        if len(self.img_files) == 0:
            raise RuntimeError(f"No image npy files found in {images_dir}")

        # Build slice index: list of (img_path, mask_path, slice_idx)
        self.index = []
        for img_path in self.img_files:
            mask_path = self.masks_dir / img_path.name.replace("image", "mask")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {img_path.name}: {mask_path}")

            # Load mask once to decide which slices to include
            m = np.load(mask_path)  # expected (D,H,W,4)
            if m.ndim != 4 or m.shape[-1] != 4:
                raise ValueError(f"Expected mask shape (D,H,W,4). Got {m.shape} in {mask_path}")

            # lesion map across channels
            lesion = (m > 0.5).any(axis=-1)  # (D,H,W)

            D = lesion.shape[self.axis]
            lesion_slices = []
            empty_slices = []

            for s in range(D):
                sl = np.take(lesion, s, axis=self.axis)
                if sl.sum() >= self.lesion_min_pixels:
                    lesion_slices.append(s)
                else:
                    empty_slices.append(s)

            if self.only_lesion_slices:
                chosen = lesion_slices
                # optionally include a few empty slices for calibration
                if self.max_empty_per_volume > 0 and len(empty_slices) > 0:
                    chosen = chosen + empty_slices[: self.max_empty_per_volume]
            else:
                chosen = list(range(D))

            for s in chosen:
                self.index.append((img_path, mask_path, s))

        if len(self.index) == 0:
            raise RuntimeError("No slices indexed — try lowering lesion_min_pixels or set only_lesion_slices=False")

        print("=" * 50)
        print(f"[BraTSNpy2D]")
        print(f"Images directory: {self.images_dir}")
        print(f"Masks directory:  {self.masks_dir}")
        print(f"Number of volumes: {len(self.img_files)}")
        print(f"Total slices indexed: {len(self.index)}")
        print(f"Mode: {self.mode}")
        print(f"Only lesion slices: {self.only_lesion_slices}")
        print("=" * 50)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_path, mask_path, s = self.index[idx]
        x = np.load(img_path).astype(np.float32)   # (D,H,W,4)
        m = np.load(mask_path).astype(np.float32)  # (D,H,W,4)

        # take slice along axis
        x_sl = np.take(x, s, axis=self.axis)       # (H,W,4) if axis=0
        m_sl = np.take(m, s, axis=self.axis)       # (H,W,4)

        # to torch: channels-first
        x_sl = torch.from_numpy(np.moveaxis(x_sl, -1, 0))  # [4,H,W]

        if self.mode == "binary":
            y = (m_sl > 0.5).any(axis=-1).astype(np.int64)  # [H,W]
            y = torch.from_numpy(y)
        elif self.mode == "multilabel":
            y = (m_sl > 0.5).astype(np.float32)             # [H,W,4]
            y = torch.from_numpy(np.moveaxis(y, -1, 0))      # [4,H,W]
        else:
            raise ValueError("mode must be 'binary' or 'multilabel'")

        meta = {"file": img_path.name, "slice": int(s)}
        return x_sl, y, meta
