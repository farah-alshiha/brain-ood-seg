from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.brats_npy_2d import BraTSNpy2D
from src.models.unet2d import UNet2D
from src.train.losses import dice_coeff_from_logits, bce_dice_loss, save_curves

@torch.no_grad()
def estimate_pos_weight(train_loader, device, max_batches=50):

    pos = 0
    neg = 0
    n = 0

    for batch_idx, (_, y, _) in enumerate(train_loader):
        y = y.to(device)
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())
        n += 1
        if n >= max_batches:
            break

    if pos == 0:
        # fallback (should not happen if lesion slices exist)
        return torch.tensor(1.0, device=device)

    pw = neg / max(pos, 1)
    return torch.tensor(float(pw), device=device)

@torch.no_grad()
def batch_confusion_from_logits(logits, y, thresh=0.5):
    """
    logits: [B,1,H,W]
    y:      [B,H,W] 0/1
    returns tp, fp, fn, tn (as python ints)
    """
    p = (torch.sigmoid(logits) > thresh).squeeze(1)  # [B,H,W] bool
    t = (y > 0)                                      # [B,H,W] bool

    tp = int((p & t).sum().item())
    fp = int((p & ~t).sum().item())
    fn = int((~p & t).sum().item())
    tn = int((~p & ~t).sum().item())
    return tp, fp, fn, tn

@torch.no_grad()
def save_tp_fp_fn_overlays(
    model, loader, out_dir: Path, device, num_samples=12, thresh=0.5, only_tumor_slices=True
):
    out_dir = Path(out_dir)
    viz_dir = out_dir / "test_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0

    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        p = (torch.sigmoid(logits) > thresh).squeeze(1)  # [B,H,W] bool
        t = (y > 0)                                      # [B,H,W] bool

        x_cpu = x.cpu().numpy()      # [B,4,H,W]
        p_cpu = p.cpu().numpy()
        t_cpu = t.cpu().numpy()

        B = x_cpu.shape[0]
        for i in range(B):
            if saved >= num_samples:
                return

            gt = t_cpu[i].astype(np.uint8)
            pred = p_cpu[i].astype(np.uint8)

            if only_tumor_slices and gt.sum() < 10:
                continue

            tp = (pred == 1) & (gt == 1)
            fp = (pred == 1) & (gt == 0)
            fn = (pred == 0) & (gt == 1)

            # background image (modality 0)
            img = x_cpu[i, 0]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # RGB overlay
            overlay = np.stack([img, img, img], axis=-1)  # [H,W,3]
            overlay[tp] = np.array([0.0, 1.0, 0.0])  # TP = green
            overlay[fp] = np.array([1.0, 0.0, 0.0])  # FP = red
            overlay[fn] = np.array([0.0, 0.0, 1.0])  # FN = blue

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap="gray")
            plt.title("Modality 0")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt, cmap="gray")
            plt.title(f"GT (px={int(gt.sum())})")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("TP/FP/FN overlay")
            plt.axis("off")

            legend_handles = [
                mpatches.Patch(color=(0.0, 1.0, 0.0), label="True Positive"),
                mpatches.Patch(color=(1.0, 0.0, 0.0), label="False Positive"),
                mpatches.Patch(color=(0.0, 0.0, 1.0), label="False Negative"),
            ]
            plt.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)

            fname = f"{meta['file'][i]}_slice{int(meta['slice'][i])}.png" if isinstance(meta, dict) else f"sample_{saved}.png"
            plt.savefig(viz_dir / fname, dpi=200, bbox_inches="tight")
            plt.close()

            saved += 1

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    bce_loss_fn,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Tuple[float, float]:
    model.train()
    tot_loss, tot_dice = 0.0, 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = bce_dice_loss(logits, y, bce_loss_fn)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = bce_dice_loss(logits, y, bce_loss_fn)
            loss.backward()
            opt.step()

        dice = dice_coeff_from_logits(logits.detach(), y)

        tot_loss += float(loss.item())
        tot_dice += float(dice.item())

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")

    return tot_loss / len(loader), tot_dice / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, device, bce_loss_fn, thresh=0.5):
    model.eval()
    tot_loss, tot_dice = 0.0, 0.0

    dice_tumor_sum = 0.0
    dice_tumor_n = 0

    TP = FP = FN = TN = 0

    empty_slices = 0
    empty_fp_slices = 0  # predicted any tumor when GT is empty

    pbar = tqdm(loader, desc="Val", leave=False)
    for x, y, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = bce_dice_loss(logits, y, bce_loss_fn)
        dice = dice_coeff_from_logits(logits, y)

        tot_loss += float(loss.item())
        tot_dice += float(dice.item())

        tp, fp, fn, tn = batch_confusion_from_logits(logits, y, thresh=thresh)
        TP += tp; FP += fp; FN += fn; TN += tn

        # tumor-only dice (exclude empty GT slices)
        has_tumor = (y.view(y.shape[0], -1).sum(dim=1) > 0)
        if has_tumor.any():
            probs = torch.sigmoid(logits).detach()
            targets = y.unsqueeze(1).float()
            num = 2.0 * (probs * targets).sum(dim=(2, 3))
            den = (probs + targets).sum(dim=(2, 3)) + 1e-6
            dice_per = (num / den).squeeze(1)  # [B]
            dice_tumor_sum += float(dice_per[has_tumor].sum().item())
            dice_tumor_n += int(has_tumor.sum().item())

        # empty-slice FP rate
        pred_any = (torch.sigmoid(logits).squeeze(1) > thresh).view(y.shape[0], -1).any(dim=1)
        gt_any = has_tumor
        empty = ~gt_any
        empty_slices += int(empty.sum().item())
        empty_fp_slices += int((empty & pred_any).sum().item())

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")

    avg_loss = tot_loss / len(loader)
    avg_dice_all = tot_dice / len(loader)
    avg_dice_tumor = (dice_tumor_sum / dice_tumor_n) if dice_tumor_n > 0 else float("nan")

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    iou       = TP / (TP + FP + FN + 1e-6)
    empty_fp_rate = empty_fp_slices / max(empty_slices, 1)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "empty_fp_rate": float(empty_fp_rate),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    }
    return avg_loss, avg_dice_all, avg_dice_tumor, metrics


@torch.no_grad()
def save_qualitative_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    num_samples: int = 8,
    threshold: float = 0.5,
    only_tumor_slices: bool = True,
) -> None:
    """
    Saves overlays for a few validation samples.
    Uses modality 0 for visualization by default.
    """
    out_dir = Path(out_dir)
    qdir = out_dir / "qual"
    qdir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0

    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()  # [B,1,H,W]
        x_cpu = x.cpu().numpy()                     # [B,4,H,W]
        y_cpu = y.numpy()                           # [B,H,W]

        B = x_cpu.shape[0]
        for i in range(B):
            if saved >= num_samples:
                return

            img = x_cpu[i, 0]  # modality 0
            gt = y_cpu[i].astype(np.uint8)
            pred = (prob[i, 0] > threshold).astype(np.uint8)

            if only_tumor_slices and gt.sum() < 10:
                continue

            plt.figure(figsize=(10, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap="gray")
            plt.title("Modality 0")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt, cmap="gray")
            plt.title(f"GT (lesion px={int(gt.sum())})")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(img, cmap="gray")
            plt.imshow(pred, alpha=0.4)
            plt.title(f"Pred (px={int(pred.sum())})")
            plt.axis("off")

            fname = f"{meta['file'][i]}_slice{int(meta['slice'][i])}.png" if isinstance(meta, dict) else f"sample_{saved}.png"
            plt.savefig(qdir / fname, dpi=200, bbox_inches="tight")
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
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--only_lesion_slices", action="store_true")
    ap.add_argument("--lesion_min_pixels", type=int, default=10)
    ap.add_argument("--max_empty_per_volume", type=int, default=0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only).")

    ap.add_argument("--save_qual", action="store_true")
    ap.add_argument("--qual_n", type=int, default=8)
    ap.add_argument("--qual_thresh", type=float, default=0.5)
    ap.add_argument("--qual_only_tumor", action="store_true")

    ap.add_argument("--test_images", type=str, default=None)
    ap.add_argument("--test_masks", type=str, default=None)

    ap.add_argument("--use_pos_weight", action="store_true", help="Use BCE pos_weight for class imbalance.")
    ap.add_argument("--pos_weight_batches", type=int, default=50, help="How many train batches to estimate pos_weight.")

    ap.add_argument("--save_test_viz", action="store_true")
    ap.add_argument("--test_viz_n", type=int, default=12)
    ap.add_argument("--test_viz_thresh", type=float, default=0.5)
    ap.add_argument("--test_viz_only_tumor", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Enable cudnn benchmark for speed (fine for training; can disable if you want strict determinism)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print("\nCreating training dataset...")
    train_ds = BraTSNpy2D(
        args.train_images,
        args.train_masks,
        mode="binary",
        only_lesion_slices=args.only_lesion_slices,
        lesion_min_pixels=args.lesion_min_pixels,
        max_empty_per_volume=args.max_empty_per_volume,
        seed=args.seed,
        verbose=True,
    )

    print("\nCreating validation dataset...")
    val_ds = BraTSNpy2D(
        args.val_images,
        args.val_masks,
        mode="binary",
        only_lesion_slices=False,
        seed=args.seed,
        verbose=True,
    )

    print("\nCreating DataLoaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    test_loader = None
    if args.test_images and args.test_masks:
        print("\nCreating test dataset/loader...")
        test_ds = BraTSNpy2D(args.test_images, args.test_masks, mode="binary", only_lesion_slices=False, seed=args.seed, verbose=True)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

    print(f"\nTrain batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # loss function (BCEWithLogits + optional pos_weight)
    if args.use_pos_weight:
        pos_weight = estimate_pos_weight(train_loader, device, max_batches=args.pos_weight_batches)
        print(f"[pos_weight] Estimated pos_weight = {pos_weight.item():.4f} (neg/pos)")
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    # First batch sanity check
    print("\nRunning first batch sanity check...")
    x0, y0, meta0 = next(iter(train_loader))
    print(f"Batch X shape: {tuple(x0.shape)} (expect [B,4,128,128])")
    print(f"Batch Y shape: {tuple(y0.shape)} (expect [B,128,128])")
    print(f"X dtype: {x0.dtype} | Y dtype: {y0.dtype} | Y unique: {torch.unique(y0)}")
    print(f"Example meta: {meta0}")
    print(f"Lesion pixels in batch: {int(y0.sum().item())}")
    print("Sanity check complete.\n")

    model = UNet2D(in_channels=4, base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: Dict[str, list] = {
        "train_loss": [],
        "train_dice": [],
        "val_precision": [],
        "val_recall": [],
        "val_iou": [],
        "val_empty_fp_rate": [],
    }

    best = -1.0
    best_epoch = -1

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, opt, device, bce_loss_fn, use_amp, scaler)
        va_loss, va_dice_all, va_dice_tumor, va_metrics = eval_one_epoch(model, val_loader, device, bce_loss_fn, thresh=0.5)

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_precision"].append(va_metrics["precision"])
        history["val_recall"].append(va_metrics["recall"])
        history["val_iou"].append(va_metrics["iou"])
        history["val_empty_fp_rate"].append(va_metrics["empty_fp_rate"])

        print(
            f"Epoch {ep:02d} | "
            f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} | "
            f"val_loss={va_loss:.4f} val_dice={va_dice_all:.4f} val_dice_tumor={va_dice_tumor:.4f} | "
            f"P={va_metrics['precision']:.3f} R={va_metrics['recall']:.3f} IoU={va_metrics['iou']:.3f} "
            f"EmptyFP={va_metrics['empty_fp_rate']:.3f}"
        )

        if va_dice_tumor == va_dice_tumor and va_dice_tumor > best:  # check not NaN
            best = float(va_dice_tumor)
            best_epoch = ep
            torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "best.pt")
    
    if args.save_test_viz and test_loader is not None:
        print("\nSaving TP/FP/FN test overlays...")
        save_tp_fp_fn_overlays(
            model=model,
            loader=test_loader,
            out_dir=out_dir,
            device=device,
            num_samples=args.test_viz_n,
            thresh=args.test_viz_thresh,
            only_tumor_slices=args.test_viz_only_tumor,
        )
        print(f"Saved test visualizations to: {out_dir/'test_viz'}")

    # Save last checkpoint too
    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "last.pt")

    # Save history + curves
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    save_curves(history, out_dir)

    # Summary
    summary = {
        "best_val_dice_tumor": best,
        "best_epoch": best_epoch,
        "final_val_dice_tumor": history["val_dice_tumor"][-1] if len(history["val_dice_tumor"]) else None,
        "final_val_dice_all": history["val_dice"][-1] if len(history["val_dice"]) else None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Saved best checkpoint: {out_dir/'best.pt'} (best_epoch={best_epoch}, best_val_dice_tumor={best:.4f})")
    print(f"Saved curves: {out_dir/'curves_loss.png'} and {out_dir/'curves_dice.png'}")
    print(f"Saved history: {out_dir/'history.json'} | summary: {out_dir/'summary.json'}")

    # Qualitative predictions
    if args.save_qual:
        print("\nSaving qualitative predictions...")
        save_qualitative_predictions(
            model=model,
            loader=val_loader,
            out_dir=out_dir,
            device=device,
            num_samples=args.qual_n,
            threshold=args.qual_thresh,
            only_tumor_slices=args.qual_only_tumor,
        )
        print(f"Saved qualitative outputs to: {out_dir/'qual'}")


if __name__ == "__main__":
    main()