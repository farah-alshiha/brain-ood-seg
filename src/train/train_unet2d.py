import argparse
from pathlib import Path
from tqdm import tqdm

import json
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.data.brats_npy_2d import BraTSNpy2D
from src.models.unet2d import UNet2D
from src.train.losses import bce_dice_loss, dice_coeff_from_logits, save_curves

import torch

@torch.no_grad()
def save_qualitative_predictions(model, loader, out_dir, device, num_samples=6):
    out_dir = Path(out_dir)
    (out_dir / "qual").mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0

    for x, y, meta in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        x_cpu = x.cpu().numpy()
        y_cpu = y.numpy()

        B = x_cpu.shape[0]
        for i in range(B):
            if saved >= num_samples:
                return

            img = x_cpu[i, 0]
            gt = y_cpu[i]
            pred = (probs[i, 0] > 0.5).astype(np.uint8)

            print("\nGT lesion pixels:", gt.sum())
            print("Pred lesion pixels:", pred.sum())

            print("\nGT shape:", gt.shape, "unique:", np.unique(gt)[:10])
            print("Pred shape:", pred.shape, "unique:", np.unique(pred)[:10])

            plt.figure(figsize=(10,3))
            plt.subplot(1,3,1); plt.imshow(img, cmap="gray"); plt.title("Modality 0"); plt.axis("off")
            plt.subplot(1,3,2); plt.imshow(gt, cmap="gray"); plt.title("GT (binary)"); plt.axis("off")
            plt.subplot(1,3,3); plt.imshow(img, cmap="gray"); plt.imshow(pred, alpha=0.4); plt.title("Pred overlay"); plt.axis("off")

            fname = f"{meta['file'][i]}_slice{meta['slice'][i]}.png" if isinstance(meta, dict) else f"sample_{saved}.png"
            plt.savefig(out_dir / "qual" / fname, dpi=200, bbox_inches="tight")
            plt.close()

            saved += 1

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", type=str, required=True)
    ap.add_argument("--train_masks", type=str, required=True)
    ap.add_argument("--val_images", type=str, required=True)
    ap.add_argument("--val_masks", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="runs/unet2d_baseline")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--only_lesion_slices", action="store_true")
    ap.add_argument("--lesion_min_pixels", type=int, default=10)
    return ap.parse_args()

def train_one_epoch(model, loader, opt, device):
    model.train()
    tot_loss, tot_dice = 0.0, 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y, _ in pbar:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = bce_dice_loss(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        dice = dice_coeff_from_logits(logits.detach(), y)

        tot_loss += loss.item()
        tot_dice += dice.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice.item():.4f}"
        })

    return tot_loss / len(loader), tot_dice / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    tot_loss, tot_dice = 0.0, 0.0

    pbar = tqdm(loader, desc="Val", leave=False)
    for x, y, _ in pbar:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = bce_dice_loss(logits, y)
        dice = dice_coeff_from_logits(logits, y)

        tot_loss += loss.item()
        tot_dice += dice.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice.item():.4f}"
        })

    return tot_loss / len(loader), tot_dice / len(loader)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating training dataset...")
    train_ds = BraTSNpy2D(
        args.train_images, args.train_masks,
        mode="binary",
        only_lesion_slices=args.only_lesion_slices,
        lesion_min_pixels=args.lesion_min_pixels
    )

    print("Creating validation dataset...")
    val_ds = BraTSNpy2D(args.val_images, args.val_masks, mode="binary", only_lesion_slices=False)

    print("\nCreating DataLoaders...")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    print("\nRunning first batch sanity check...")

    x, y, meta = next(iter(train_loader))

    print(f"Batch X shape: {x.shape}")
    print(f"Batch Y shape: {y.shape}")
    print(f"Sample meta: {meta}")
    print(f"X dtype: {x.dtype}")
    print(f"Y unique values: {torch.unique(y)}")

    print("Sanity check complete.\n")

    history = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
    }

    model = UNet2D(in_channels=4, base=args.base).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, opt, args.device)
        va_loss, va_dice = eval_one_epoch(model, val_loader, args.device)

        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} "
              f"| val_loss={va_loss:.4f} val_dice={va_dice:.4f}")
        
        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(va_loss)
        history["val_dice"].append(va_dice)

        if va_dice > best:
            best = va_dice
            ckpt = out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    save_curves(history, out_dir / "curves")
    print(f"Saved training curves to: {out_dir}")

    save_qualitative_predictions(model, val_loader, out_dir, args.device, num_samples=8)
    print(f"Saved qualitative predictions to: {out_dir/'qual'}")

if __name__ == "__main__":
    main()