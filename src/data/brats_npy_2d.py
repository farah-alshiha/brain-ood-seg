from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSNpy2D(Dataset):
    """
    BraTS .npy 3D volumes -> 2D slice dataset.

    Expected file format:
      - image: (D, H, W, 4) float (already normalized to [0, 1] in your dataset)
      - mask : (D, H, W, 4) float/bool, where channel 0 is (mostly) foreground/brain mask
              and channels 1..3 correspond to tumor-related regions (values {0,1})

    Modes:
      - binary: returns tumor union mask over channels 1..3 -> y shape (H, W), dtype long (0/1)
      - multilabel: returns tumor channels 1..3 -> y shape (3, H, W), dtype float (0/1)
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        mode: str = "binary",              # "binary" or "multilabel"
        axis: int = 0,                     # slice axis in (D,H,W,4); 0 means axial (D)
        only_lesion_slices: bool = True,   # if True, index only slices with tumor pixels
        lesion_min_pixels: int = 10,       # minimum tumor pixels to count as "lesion slice"
        max_empty_per_volume: int = 0,     # optionally add some empty slices per volume
        seed: int = 42,                    # for deterministic empty-slice sampling
        validate_files: bool = True,       # sanity-check a few files at init
        cache_volumes: bool = False,       # optional caching (safe in single-worker; see note)
        verbose: bool = True,
        augment: bool = False,
        aug_p: float = 0.8,
        flip_p: float = 0.5,
        rot90_p: float = 0.5,
        intensity_p: float = 0.8,
        noise_p: float = 0.5,
        gamma_p: float = 0.5,
        noise_std: float = 0.03,
        gamma_range: tuple[float, float] = (0.7, 1.5),
        intensity_range: tuple[float, float] = (0.9, 1.1),
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mode = mode
        self.axis = axis
        self.only_lesion_slices = only_lesion_slices
        self.lesion_min_pixels = int(lesion_min_pixels)
        self.max_empty_per_volume = int(max_empty_per_volume)
        self.seed = int(seed)
        self.validate_files = bool(validate_files)
        self.cache_volumes = bool(cache_volumes)
        self.verbose = bool(verbose)
        self.augment = augment
        self.aug_p = float(aug_p)
        self.flip_p = float(flip_p)
        self.rot90_p = float(rot90_p)
        self.intensity_p = float(intensity_p)
        self.noise_p = float(noise_p)
        self.gamma_p = float(gamma_p)
        self.noise_std = float(noise_std)
        self.gamma_range = gamma_range
        self.intensity_range = intensity_range
        self._rng = np.random.default_rng(self.seed)

        if self.mode not in {"binary", "multilabel"}:
            raise ValueError("mode must be 'binary' or 'multilabel'")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"masks_dir not found: {self.masks_dir}")

        self.img_files = sorted([p for p in self.images_dir.iterdir() if p.suffix == ".npy" and "image" in p.name])
        if len(self.img_files) == 0:
            raise RuntimeError(f"No image npy files found in {self.images_dir}")

        # Optional lightweight validation on first few pairs
        if self.validate_files:
            for img_path in self.img_files[:3]:
                mask_path = self.masks_dir / img_path.name.replace("image", "mask")
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing mask for {img_path.name}: {mask_path}")
                x = np.load(img_path, mmap_mode="r")
                m = np.load(mask_path, mmap_mode="r")
                if x.ndim != 4 or x.shape[-1] != 4:
                    raise ValueError(f"Expected image shape (D,H,W,4). Got {x.shape} in {img_path}")
                if m.ndim != 4 or m.shape[-1] != 4:
                    raise ValueError(f"Expected mask shape (D,H,W,4). Got {m.shape} in {mask_path}")

        # Build slice index: list of (img_path, mask_path, slice_idx)
        self.index: List[Tuple[Path, Path, int]] = []
        rng = np.random.default_rng(self.seed)

        for img_path in self.img_files:
            mask_path = self.masks_dir / img_path.name.replace("image", "mask")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {img_path.name}: {mask_path}")

            # Load mask once per volume to build slice list (mmap keeps it light-ish)
            m = np.load(mask_path, mmap_mode="r")  # expected (D,H,W,4)

            # Tumor union across channels 1..3 (exclude channel 0 which is foreground/brain mask)
            tumor_union = (m[..., 1:] > 0.5).any(axis=-1)  # (D,H,W) boolean

            D = tumor_union.shape[self.axis]
            lesion_slices: List[int] = []
            empty_slices: List[int] = []

            for s in range(D):
                sl = np.take(tumor_union, s, axis=self.axis)  # (H,W) boolean
                if int(sl.sum()) >= self.lesion_min_pixels:
                    lesion_slices.append(s)
                else:
                    empty_slices.append(s)

            if self.only_lesion_slices:
                chosen = list(lesion_slices)
                if self.max_empty_per_volume > 0 and len(empty_slices) > 0:
                    rng.shuffle(empty_slices)
                    chosen.extend(empty_slices[: self.max_empty_per_volume])
            else:
                chosen = list(range(D))

            for s in chosen:
                self.index.append((img_path, mask_path, int(s)))

        if len(self.index) == 0:
            raise RuntimeError(
                "No slices indexed — try lowering lesion_min_pixels or set only_lesion_slices=False."
            )

        # Optional in-memory cache (NOTE: in multi-worker DataLoader, each worker gets its own dataset copy)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        if self.verbose:
            print("=" * 60)
            print("[BraTSNpy2D] Dataset initialized")
            print(f"Images directory:      {self.images_dir}")
            print(f"Masks directory:       {self.masks_dir}")
            print(f"Number of volumes:     {len(self.img_files)}")
            print(f"Total slices indexed:  {len(self.index)}")
            print(f"Mode:                 {self.mode}")
            print(f"Axis:                 {self.axis}")
            print(f"Only lesion slices:   {self.only_lesion_slices}")
            print(f"lesion_min_pixels:    {self.lesion_min_pixels}")
            print(f"max_empty_per_volume: {self.max_empty_per_volume}")
            print(f"cache_volumes:        {self.cache_volumes}")
            print("=" * 60)

    def __len__(self) -> int:
        return len(self.index)

    def _load_pair(self, img_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image/mask pair. Uses mmap by default for speed/memory,
        optionally caches full arrays if cache_volumes=True.
        """
        key = img_path.name
        if self.cache_volumes and key in self._cache:
            return self._cache[key]

        # mmap_mode speeds repeated slice access without reading entire file each time
        x = np.load(img_path, mmap_mode="r")  # (D,H,W,4)
        m = np.load(mask_path, mmap_mode="r")  # (D,H,W,4)

        # If caching, materialize to RAM as float32 to avoid repeated disk reads
        if self.cache_volumes:
            x = np.asarray(x, dtype=np.float32)
            m = np.asarray(m, dtype=np.float32)
            self._cache[key] = (x, m)

        return x, m

    def __getitem__(self, idx: int):
        img_path, mask_path, s = self.index[idx]
        x, m = self._load_pair(img_path, mask_path)

        # Take slice along axis; for typical (D,H,W,4) and axis=0 => (H,W,4)
        x_sl = np.take(x, s, axis=self.axis)
        m_sl = np.take(m, s, axis=self.axis)

        # Sanity checks (cheap)
        if x_sl.ndim != 3 or x_sl.shape[-1] != 4:
            raise ValueError(f"Expected image slice (H,W,4). Got {x_sl.shape} from {img_path.name} slice {s}")
        if m_sl.ndim != 3 or m_sl.shape[-1] != 4:
            raise ValueError(f"Expected mask slice (H,W,4). Got {m_sl.shape} from {mask_path.name} slice {s}")

        # Convert image to torch [4,H,W] float32
        x_sl = np.asarray(x_sl, dtype=np.float32)
        x_t = torch.from_numpy(np.moveaxis(x_sl, -1, 0))  # [4,H,W]
        if x_t.ndim != 3 or x_t.shape[0] != 4:
            raise ValueError(f"Expected x tensor [4,H,W]. Got {tuple(x_t.shape)} from {img_path.name} slice {s}")

        # Targets
        m_sl = np.asarray(m_sl)  # may be float64 if not cached; keep as numpy
        if self.mode == "binary":
            # Union over channels 1..3 (exclude channel 0 foreground/brain mask)
            y = (m_sl[..., 1:] > 0.5).any(axis=-1).astype(np.int64)  # (H,W)
            x_np = np.moveaxis(x_sl, -1, 0).astype(np.float32)       # [4,H,W]

            if self.augment:
                x_np, y = self._augment_xy(x_np, y)

            x_t = torch.from_numpy(x_np)
            y_t = torch.from_numpy(y)
            if y_t.ndim != 2:
                raise ValueError(f"Binary y expected [H,W]. Got {tuple(y_t.shape)} from {mask_path.name} slice {s}")
        else:
            # Multilabel: tumor channels only (3, H, W)
            y = (m_sl[..., 1:] > 0.5).astype(np.float32)  # (H,W,3)
            y_t = torch.from_numpy(np.moveaxis(y, -1, 0))  # (3,H,W)
            if y_t.ndim != 3 or y_t.shape[0] != 3:
                raise ValueError(f"Multilabel y expected [3,H,W]. Got {tuple(y_t.shape)} from {mask_path.name} slice {s}")

        meta = {"file": img_path.name, "slice": int(s)}
        return x_t, y_t, meta
    
    def _augment_xy(self, x: np.ndarray, y: np.ndarray):
        """
        x: [4,H,W] float32
        y: [H,W] int64 (binary) OR [3,H,W] float32 (multilabel)
        Returns augmented (x,y) with spatial ops applied to both, intensity ops to x only.
        """
        rng = self._rng

        if rng.random() > self.aug_p:
            return x, y

        # Random horizontal flip
        if rng.random() < self.flip_p:
            x = x[:, :, ::-1].copy()
            if y.ndim == 2:
                y = y[:, ::-1].copy()
            else:
                y = y[:, :, ::-1].copy()

        # Random vertical flip
        if rng.random() < self.flip_p:
            x = x[:, ::-1, :].copy()
            if y.ndim == 2:
                y = y[::-1, :].copy()
            else:
                y = y[:, ::-1, :].copy()

        # Random 90-degree rotations (0,1,2,3)
        if rng.random() < self.rot90_p:
            k = int(rng.integers(0, 4))
            if k != 0:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                if y.ndim == 2:
                    y = np.rot90(y, k=k, axes=(0, 1)).copy()
                else:
                    y = np.rot90(y, k=k, axes=(1, 2)).copy()

        if rng.random() < self.intensity_p:
            scale = rng.uniform(self.intensity_range[0], self.intensity_range[1])
            x = x * scale

        if rng.random() < self.gamma_p:
            g = rng.uniform(self.gamma_range[0], self.gamma_range[1])
            # keep in [0,1] assumption
            x = np.clip(x, 0.0, 1.0)
            x = x ** g

        if rng.random() < self.noise_p:
            n = rng.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
            x = x + n

        x = np.clip(x, 0.0, 1.0)
        return x, y