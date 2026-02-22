import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.brats_npy_2d import BraTSNpy2D
from src.models.unet2d import UNet2D
from src.train.losses import bce_dice_loss, dice_coeff_from_logits

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
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = bce_dice_loss(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        tot_loss += loss.item()
        tot_dice += dice_coeff_from_logits(logits.detach(), y).item()

    return tot_loss / len(loader), tot_dice / len(loader)

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    tot_loss, tot_dice = 0.0, 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = bce_dice_loss(logits, y)
        tot_loss += loss.item()
        tot_dice += dice_coeff_from_logits(logits, y).item()
    return tot_loss / len(loader), tot_dice / len(loader)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BraTSNpy2D(
        args.train_images, args.train_masks,
        mode="binary",
        only_lesion_slices=args.only_lesion_slices,
        lesion_min_pixels=args.lesion_min_pixels
    )
    val_ds = BraTSNpy2D(args.val_images, args.val_masks, mode="binary", only_lesion_slices=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = UNet2D(in_channels=4, base=args.base).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, opt, args.device)
        va_loss, va_dice = eval_one_epoch(model, val_loader, args.device)

        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} "
              f"| val_loss={va_loss:.4f} val_dice={va_dice:.4f}")

        if va_dice > best:
            best = va_dice
            ckpt = out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)

if __name__ == "__main__":
    main()