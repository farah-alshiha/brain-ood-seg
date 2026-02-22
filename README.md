# Brain OOD Segmentation (BraTS 2D Baseline)

This repository implements a clean, modular 2D multimodal brain tumor segmentation baseline using BraTS `.npy` volumes.

The goal of this project is to build a strong segmentation foundation that will later be extended to:

- OOD (Out-of-Distribution) detection  
- OOD-gated test-time adaptation  
- Cross-modality contrastive learning  
- Near-OOD evaluation (BraTS → REMBRANDT)  

The current version provides a reproducible, path-agnostic UNet2D baseline.

---

## Repository Structure
```
brain-ood-seg/
│
├── src/
│ ├── data/
│ │ ├── init.py
│ │ └── brats_npy_2d.py
│ │
│ ├── models/
│ │ ├── init.py
│ │ └── unet2d.py
│ │
│ └── train/
│ └── train_unet2d.py
│
├── requirements.txt
└── README.md
```

---

## Dataset Format

This implementation expects BraTS data stored as:
```
train/
images/
image_001.npy
image_002.npy
masks/
mask_001.npy
mask_002.npy

val/
images/
masks/
```

Each `.npy` file must contain:

**Image:**
- Shape: `(128, 128, 128, 4)`
- 4 MRI modalities (T1, T1ce, T2, FLAIR)
- Normalized to range `[0, 1]`

**Mask:**
- Shape: `(128, 128, 128, 4)`
- One-hot encoded tumor channels
- Values `{0, 1}`

The current baseline converts masks to **binary tumor segmentation**.

---

## Installation

Clone the repository:

```
git clone https://github.com/farah-alshiha/brain-ood-seg.git
cd brain-ood-seg
```

Install dependencies:
```
pip install -r requirements.txt
```

## Training (Local Machine)

Run:
```
python3 -m src.train.train_unet2d \
  --train_images /path/to/train/images \
  --train_masks  /path/to/train/masks \
  --val_images   /path/to/val/images \
  --val_masks    /path/to/val/masks \
  --out_dir runs/unet2d_baseline \
  --epochs 10 \
  --batch 16 \
  --lr 1e-3 \
  --only_lesion_slices
```
---

## Running in Google Colab

Clone repository:
```
!git clone https://github.com/farah-alshiha/brain-ood-seg.git
%cd brain-ood-seg
```

Install dependencies:
```
!pip install -r requirements.txt
```

Train:
```
!python3 -m src.train.train_unet2d \
  --train_images "/path_to_brats/train/images" \ # change paths for first four args
  --train_masks  "/path_to_brats/train/masks" \
  --val_images   "/path_to_brats/val/images" \
  --val_masks    "/path_to_brats/val/masks" \
  --out_dir runs/unet2d_baseline \
  --epochs 10 \
  --batch 16 \
  --lr 1e-3 \
  --only_lesion_slices
```
---

## Model Details

Baseline model:

* 2D U-Net
* 4-channel input
* 1-channel binary output
* BCE + Dice loss
* AdamW optimizer

Best checkpoint is saved to:

```
runs/unet2d_baseline/best.pt
```
---

## Notes

* Dataset files are excluded via .gitignore.
* Do not commit .npy, .nii, or large data files.
* This baseline is intentionally simple and clean before adding OOD components.

---

## Next Steps

* Planned extensions:
* Image-level OOD detection (entropy / energy)
* OOD-gated test-time adaptation
* Cross-modality contrastive alignment
* Near-OOD evaluation (BraTS → REMBRANDT)

---

## Author

Farah Alshiha @ BRAIN Lab — KFUPM
