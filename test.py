"""
AC-3 Test Script: Per-image evaluation with Dice/IoU/HD95.

Key protocol requirements:
- batch_size=1 (per-image evaluation)
- Dice, IoU, HD95 computed per-image, then averaged
- CSV output with per-image metrics + mean summary row
- Test set NEVER seen during training or checkpoint selection
- Loads checkpoint selected by best val IoU during training

Usage:
  python test.py --data_dir ./data --dataset busi --ckpt ./runs/.../best_ep050_iou0.8500_dice0.9100.pth
"""

import os
import sys
import csv
import math
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset import MedicalSegDataset
from dpt import DPT
from train import load_backbone, build_model, get_device

# Re-use train.py's load_backbone (handles DINOv2 and DINOv3)
from utils import (dice_numpy, iou_numpy, hd95_numpy,
                    precision_numpy, recall_numpy, specificity_numpy, accuracy_numpy)


@torch.no_grad()
def run_test(model, loader, device, dice_thr=0.5, csv_path=None, safe_floor=False):
    """Run per-image evaluation on test set.

    Args:
        model: trained model
        loader: test dataloader (batch_size=1)
        device: torch device
        dice_thr: binarization threshold
        csv_path: if set, write per-image metrics CSV
        safe_floor: if True, take per-pixel max(probs_prompt, probs_noprompt)
            so the DGF prompt can only boost, never suppress (direction 1:
            inference-time negative-transfer safety switch).

    Returns:
        (mean_dice, mean_iou, mean_hd95)
    """
    model.eval()
    rows = []
    dices, ious, hd95s = [], [], []
    precisions, recalls, specificities, accuracies = [], [], [], []

    pbar = tqdm(loader, desc='[Test]')
    for inputs, targets, names in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        probs = torch.sigmoid(logits)
        if safe_floor:
            model.set_prompt(False)
            logits_floor = model(inputs)
            model.set_prompt(True)
            probs = torch.maximum(probs, torch.sigmoid(logits_floor))

        B = inputs.size(0)
        for b in range(B):
            gt = targets[b, 0].detach().cpu().numpy() > 0.5
            pr = probs[b, 0].detach().cpu().numpy() > dice_thr

            dsc = dice_numpy(pr, gt)
            iou = iou_numpy(pr, gt)
            hd = hd95_numpy(pr, gt)
            prec = precision_numpy(pr, gt)
            rec = recall_numpy(pr, gt)
            spec = specificity_numpy(pr, gt)
            acc = accuracy_numpy(pr, gt)

            dices.append(dsc)
            ious.append(iou)
            hd95s.append(hd)
            precisions.append(prec)
            recalls.append(rec)
            specificities.append(spec)
            accuracies.append(acc)
            rows.append({
                'id': names[b] if isinstance(names, (list, tuple)) else names,
                'dice': dsc,
                'iou': iou,
                'hd95': hd,
                'precision': prec,
                'recall': rec,
                'specificity': spec,
                'accuracy': acc,
            })

        finite_hd = [x for x in hd95s if math.isfinite(x)]
        pbar.set_postfix(
            mDice=f'{np.mean(dices):.4f}',
            mIoU=f'{np.mean(ious):.4f}',
            mHD95=f'{np.mean(finite_hd) if finite_hd else 0.0:.2f}',
        )

    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    finite_hd = [x for x in hd95s if math.isfinite(x)]
    mean_hd95 = float(np.mean(finite_hd)) if finite_hd else float('inf')
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_specificity = float(np.mean(specificities)) if specificities else 0.0
    mean_accuracy = float(np.mean(accuracies)) if accuracies else 0.0

    n_inf = sum(1 for x in hd95s if not math.isfinite(x))

    print('=' * 60)
    print(f'[Test Summary]')
    print(f'  Dice        = {mean_dice:.4f}')
    print(f'  IoU         = {mean_iou:.4f}')
    print(f'  HD95        = {mean_hd95:.2f}  (excluded inf: {n_inf})')
    print(f'  Precision   = {mean_precision:.4f}')
    print(f'  Recall      = {mean_recall:.4f}')
    print(f'  Specificity = {mean_specificity:.4f}')
    print(f'  Accuracy    = {mean_accuracy:.4f}')
    print(f'  Total images: {len(dices)}')
    print('=' * 60)

    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'id', 'dice', 'iou', 'hd95', 'precision', 'recall', 'specificity', 'accuracy',
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            writer.writerow({
                'id': 'MEAN',
                'dice': mean_dice,
                'iou': mean_iou,
                'hd95': f'{mean_hd95:.2f} (excl inf: {n_inf})',
                'precision': mean_precision,
                'recall': mean_recall,
                'specificity': mean_specificity,
                'accuracy': mean_accuracy,
            })
        print(f'[CSV] {csv_path}')

    return mean_dice, mean_iou, mean_hd95


def load_checkpoint(model, ckpt_path, device):
    """Flexible checkpoint loading."""
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and 'state_dict' in obj:
        state = obj['state_dict']
        info = {k: obj.get(k) for k in ['epoch', 'val_iou', 'val_dice', 'val_loss']}
    else:
        state = obj
        info = {}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[Warn] Missing keys: {missing}')
    if unexpected:
        print(f'[Warn] Unexpected keys: {unexpected}')
    return info


def main():
    parser = argparse.ArgumentParser(description='AC-3: SegDINO + DGF Testing')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='busi')
    parser.add_argument('--img_dir_name', type=str, default='images')
    parser.add_argument('--label_dir_name', type=str, default='masks')
    parser.add_argument('--img_size', type=int, default=252)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)

    # Model (must match training config)
    # Model (must match training config)
    parser.add_argument('--dino_version', type=int, default=2)
    parser.add_argument('--dino_repo', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'segdino', 'dinov3-main', 'dinov3-main'))
    parser.add_argument('--dino_ckpt', type=str, default='')
    parser.add_argument('--dino_size', type=str, default='s', choices=['s', 'b'])
    parser.add_argument('--dino_weight', type=str, default='',
                        help='Path to DINOv2 pretrained weights (.pth)')
    parser.add_argument('--decoder_dim', type=int, default=256)
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=14)

    # DGF (must match training)
    parser.add_argument('--use_dgf', type=int, default=1)
    parser.add_argument('--dgf_ratio', type=float, default=0.5)
    parser.add_argument('--dgf_adapter', type=str, default='mlp', choices=['mlp', 'dualkernel', 'cavpt', 'pro'],
                        help='must match training: mlp, dualkernel, cavpt, or pro')

    # L3 (must match training)
    parser.add_argument('--use_l3', type=int, default=0,
                        help='1 = LeanLinear (L3Seg) was applied during training')
    parser.add_argument('--l3_rank', type=int, default=8,
                        help='low-rank dimension for LeanLinear (must match training)')
    parser.add_argument('--l3_blocks', type=str, default='all', choices=['all', 'odd', 'even', 'head', 'tail'],
                        help='which blocks LeanLinear wraps (must match training)')
    parser.add_argument('--l3_mode', type=str, default='both', choices=['both', 'attn', 'mlp'],
                        help='which branch LeanLinear wraps (must match training)')
    parser.add_argument('--use_stem', type=int, default=0,
                        help='1 = StemAdapter was applied during training (must match)')

    # UAG (must match training)
    parser.add_argument('--use_uag', type=int, default=0,
                        help='1 = use Uncertainty-Adaptive Gate at l4')

    # LLAR (must match training)
    parser.add_argument('--use_llar', type=int, default=0,
                        help='1 = decoder-side low-level rescue branch (module B)')

    # Test
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dice_thr', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='./test_results')

    # Direction 1: inference-time negative-transfer safety floor
    parser.add_argument('--use_safe_floor', type=int, default=0,
                        help='1 = per-pixel max(prompt, no-prompt) floor '
                             '(prompt can only boost, never suppress)')

    args = parser.parse_args()

    device = get_device()
    print(f'[Device] {device}')
    print(f'[Checkpoint] {args.ckpt}')

    # ---- Test dataset (split_seed=42 FIXED) ----
    split_seed = 42
    test_dataset = MedicalSegDataset(
        root=os.path.join(args.data_dir, args.dataset),
        split='test', split_seed=split_seed, img_size=(args.img_size, args.img_size),
        img_dir_name=args.img_dir_name, label_dir_name=args.label_dir_name,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )
    print(f'[Dataset] test={len(test_dataset)} images (split_seed={split_seed})')

    # ---- Model ----
    backbone = load_backbone(args)

    # Build a temporary args-like object for build_model
    class ModelArgs:
        pass
    model_args = ModelArgs()
    model_args.use_dgf = args.use_dgf
    model_args.dgf_ratio = args.dgf_ratio
    model_args.dino_size = args.dino_size
    model_args.num_classes = args.num_classes
    model_args.decoder_dim = args.decoder_dim
    model_args.patch_size = args.patch_size
    model_args.use_bn = args.use_bn
    model_args.use_uag = args.use_uag
    model_args.use_llar = args.use_llar
    model_args.dgf_adapter = args.dgf_adapter
    model_args.use_l3 = args.use_l3
    model_args.l3_rank = args.l3_rank
    model_args.use_stem = args.use_stem
    model_args.l3_blocks = args.l3_blocks
    model_args.l3_mode = args.l3_mode
    model_args.dino_weight = args.dino_weight
    model_args.dino_version = args.dino_version

    model = build_model(model_args, backbone)
    model = model.to(device)

    # Load checkpoint
    info = load_checkpoint(model, args.ckpt, device)
    if info:
        print(f'[CKPT Info] epoch={info.get("epoch")}, val_iou={info.get("val_iou"):.4f}, '
              f'val_dice={info.get("val_dice"):.4f}')

    # ---- Output ----
    dgf_flag = f'dgf{args.dgf_ratio}' if args.use_dgf else 'nodgf'
    uag_flag = 'uag' if getattr(args, 'use_uag', 0) else 'nouag'
    llar_flag = 'llar' if getattr(args, 'use_llar', 0) else 'nollar'
    adapter_flag = getattr(args, 'dgf_adapter', 'mlp')
    l3_flag = f'l3r{getattr(args, "l3_rank", 8)}' if getattr(args, 'use_l3', 0) else 'nol3'
    safe_flag = 'safefloor' if getattr(args, 'use_safe_floor', 0) else 'raw'
    ckpt_stem = Path(args.ckpt).stem
    save_dir = Path(args.save_dir) / f'{args.dataset}_{dgf_flag}_{adapter_flag}_{uag_flag}_{llar_flag}_{l3_flag}_{safe_flag}'
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f'test_metrics_{ckpt_stem}.csv'

    # ---- Run ----
    mean_dice, mean_iou, mean_hd95 = run_test(
        model, test_loader, device, dice_thr=args.dice_thr, csv_path=str(csv_path),
        safe_floor=bool(args.use_safe_floor),
    )

    return mean_dice, mean_iou, mean_hd95


if __name__ == '__main__':
    main()
