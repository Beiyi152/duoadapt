"""
Metrics for medical image segmentation evaluation.

All metrics are PER-IMAGE (not batch-aggregated) as required by the experiment guide.
Dice, IoU, and HD95 are computed via numpy for reproducibility.
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Training-time metrics (torch, batch-level, for progress bar only)
# ---------------------------------------------------------------------------

def dice_coeff(pred_logits, target, smooth=1e-5):
    """Torch Dice for training progress display (NOT final evaluation)."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_coeff(pred_logits, target, smooth=1e-5):
    """Torch IoU for training progress display (NOT final evaluation)."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ---------------------------------------------------------------------------
# Evaluation-time metrics (numpy, per-image, for test CSVs and paper numbers)
# ---------------------------------------------------------------------------

def dice_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    """Per-image binary Dice coefficient."""
    inter = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    s = pred_bin.sum(dtype=np.float64) + tgt_bin.sum(dtype=np.float64)
    return float((2.0 * inter + eps) / (s + eps))


def iou_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    """Per-image binary IoU / Jaccard index."""
    inter = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    union = np.logical_or(pred_bin, tgt_bin).sum(dtype=np.float64)
    return float((inter + eps) / (union + eps))


def precision_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    """Per-image Precision = TP / (TP + FP)."""
    TP = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    FP = np.logical_and(pred_bin, np.logical_not(tgt_bin)).sum(dtype=np.float64)
    return float((TP + eps) / (TP + FP + eps))


def recall_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    """Per-image Recall / Sensitivity = TP / (TP + FN)."""
    TP = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    FN = np.logical_and(np.logical_not(pred_bin), tgt_bin).sum(dtype=np.float64)
    return float((TP + eps) / (TP + FN + eps))


def specificity_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    """Per-image Specificity = TN / (TN + FP)."""
    TN = np.logical_and(np.logical_not(pred_bin), np.logical_not(tgt_bin)).sum(dtype=np.float64)
    FP = np.logical_and(pred_bin, np.logical_not(tgt_bin)).sum(dtype=np.float64)
    return float((TN + eps) / (TN + FP + eps))


def accuracy_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    """Per-image Accuracy = (TP + TN) / (TP + TN + FP + FN)."""
    TP = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    TN = np.logical_and(np.logical_not(pred_bin), np.logical_not(tgt_bin)).sum(dtype=np.float64)
    total = pred_bin.size
    return float((TP + TN + eps) / (total + eps))


def hd95_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray) -> float:
    """Per-image 95th-percentile Hausdorff Distance (pixel units).

    Uses scipy.ndimage.distance_transform_edt for distance computation.
    Returns float('inf') if only one mask is empty.
    Returns 0.0 if both masks are empty.
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion

    pred_has = pred_bin.any()
    tgt_has = tgt_bin.any()

    if not pred_has and not tgt_has:
        return 0.0
    if pred_has != tgt_has:
        return float('inf')

    # Extract boundaries
    Pb = np.logical_and(pred_bin, np.logical_not(binary_erosion(pred_bin)))
    Tb = np.logical_and(tgt_bin, np.logical_not(binary_erosion(tgt_bin)))

    if Pb.max() == 0:
        Pb = pred_bin.astype(bool)
    if Tb.max() == 0:
        Tb = tgt_bin.astype(bool)

    dist_to_T = distance_transform_edt(np.logical_not(Tb))
    dist_to_P = distance_transform_edt(np.logical_not(Pb))

    Py, Px = np.where(Pb)
    Ty, Tx = np.where(Tb)
    if len(Py) == 0 or len(Ty) == 0:
        return float('inf')

    d_all = np.concatenate([dist_to_T[Py, Px], dist_to_P[Ty, Tx]], axis=0)
    if d_all.size == 0:
        return float('inf')
    return float(np.percentile(d_all, 95))


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics_per_image(logits: torch.Tensor, targets: torch.Tensor,
                              dice_thr: float = 0.5) -> list:
    """Compute per-image Dice, IoU, HD95 from a batch of logits.

    Args:
        logits: (B, 1, H, W) raw model output
        targets: (B, 1, H, W) binary ground truth
        dice_thr: threshold for binarization

    Returns:
        List of dicts with keys: dice, iou, hd95
    """
    probs = torch.sigmoid(logits)
    B = logits.size(0)
    results = []

    for b in range(B):
        gt = targets[b, 0].detach().cpu().numpy() > 0.5
        pr = probs[b, 0].detach().cpu().numpy() > dice_thr

        results.append({
            'dice': dice_numpy(pr, gt),
            'iou': iou_numpy(pr, gt),
            'hd95': hd95_numpy(pr, gt),
            'precision': precision_numpy(pr, gt),
            'recall': recall_numpy(pr, gt),
            'specificity': specificity_numpy(pr, gt),
            'accuracy': accuracy_numpy(pr, gt),
        })

    return results
