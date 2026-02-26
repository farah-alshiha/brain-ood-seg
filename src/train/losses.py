import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def dice_coeff_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits:  [B,1,H,W]
    targets: [B,H,W] (0/1)
    """
    probs = torch.sigmoid(logits)
    targets = targets.unsqueeze(1).float()
    num = 2.0 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return (num / den).mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 0.5) -> torch.Tensor:
    targets_f = targets.unsqueeze(1).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets_f)
    dice = 1.0 - dice_coeff_from_logits(logits, targets)
    return bce + dice_weight * dice

def save_curves(history, out_path_prefix):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(str(out_path_prefix) + "_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history["train_dice"], label="train_dice")
    plt.plot(history["val_dice"], label="val_dice")
    plt.title("Dice vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.grid(True)
    plt.legend()
    plt.savefig(str(out_path_prefix) + "_dice.png", dpi=200, bbox_inches="tight")
    plt.close()