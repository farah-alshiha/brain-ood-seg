import torch
import torch.nn.functional as F

def dice_coeff_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.unsqueeze(1).float()
    num = 2 * (probs * targets).sum(dim=(2,3))
    den = (probs + targets).sum(dim=(2,3)) + eps
    return (num / den).mean()

def bce_dice_loss(logits, targets, dice_weight=0.5):
    targets_f = targets.unsqueeze(1).float()
    bce = F.binary_cross_entropy_with_logits(logits, targets_f)
    dice = 1 - dice_coeff_from_logits(logits, targets)
    return bce + dice_weight * dice