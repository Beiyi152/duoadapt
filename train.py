"""
AC-3 Training Script: SegDINO + DGF + UAG.

Supports DINOv2 (public, auto-download) and DINOv3 (gated, manual weights).

Key protocol requirements:
- Split seed=42 (FIXED) for 70/15/15 train/val/test
- Training seeds: 42, 1457, 911 (multi-seed initialization)
- Checkpoint selection: best val Dice
- Test set NEVER seen during training or checkpoint selection
- Backbone: frozen DINOv2/DINOv3 ViT; DGF adapters + decoder trainable
- UAG: uncertainty-adaptive gate at decoder l4 (variance-gated refinement)
- Per-image metrics for evaluation

Usage:
  # Baseline (no DGF)
  python train.py --dataset busi --use_dgf 0 --seed 42

  # DGF only
  python train.py --dataset busi --use_dgf 1 --dgf_ratio 0.5 --seed 42

  # DGF + UAG (GO gate)
  python train.py --dataset busi --use_dgf 1 --use_uag 1 --seed 42
"""

import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MedicalSegDataset
from dpt import DPT
from utils import dice_coeff, iou_coeff


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_backbone(args):
    """Load DINOv2 backbone from local dinov2 package (no network needed).

    Uses local dinov2/ package shipped with the release.
    Supports --dino_version 2 (default, local package) or 3 (gated, needs --dino_ckpt).
    """
    if getattr(args, 'dino_version', 2) == 3:
        dino_repo = args.dino_repo
        model_name = 'dinov3_vitb16' if args.dino_size == 'b' else 'dinov3_vits16'
        backbone = torch.hub.load(dino_repo, model_name, source='local', weights=args.dino_ckpt)
    else:
        from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14
        weight_path = getattr(args, 'dino_weight', '')
        if args.dino_size == 'b':
            backbone = dinov2_vitb14(pretrained=True, weights=weight_path)
        else:
            backbone = dinov2_vits14(pretrained=True, weights=weight_path)
    try:
        backbone.set_is_grad_checkpointing(False)
    except AttributeError:
        pass  # older dinov2 version without this method
    return backbone


def build_model(args, backbone):
    """Build DPT model with optional DGF adapters and UAG."""
    encoder_size_map = {'s': 'small', 'b': 'base'}
    dino_version = getattr(args, 'dino_version', 2)
    model = DPT(
        encoder_size=encoder_size_map.get(args.dino_size, 'small'),
        nclass=args.num_classes,
        decoder_channels=args.decoder_dim,
        patch_size=args.patch_size,
        use_bn=args.use_bn,
        backbone=backbone,
        use_dgf=args.use_dgf,
        dgf_ratio=args.dgf_ratio,
        dino_version=dino_version,
        use_uag=getattr(args, 'use_uag', False),
        use_llar=getattr(args, 'use_llar', False),
        dgf_adapter=getattr(args, 'dgf_adapter', 'mlp'),
        use_l3=getattr(args, 'use_l3', False),
        l3_rank=getattr(args, 'l3_rank', 8),
        use_stem=getattr(args, 'use_stem', False),
        l3_blocks=getattr(args, 'l3_blocks', 'all'),
        l3_mode=getattr(args, 'l3_mode', 'both'),
    )
    if not args.use_dgf and not args.use_l3 and not getattr(args, 'use_stem', False):
        model.lock_backbone()
    return model


def count_all_params(model, backbone) -> dict:
    """Count model parameters by component."""
    backbone_total = sum(p.numel() for p in backbone.parameters())
    adapter_params = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        'backbone': backbone_total,
        'dgf': adapter_params,
        'decoder': decoder_params,
        'total_trainable': trainable,
        'total_frozen': frozen,
    }


@torch.no_grad()
def evaluate(model, val_loader, device, criterion):
    """Evaluate on validation set. Returns per-image metrics."""
    model.eval()
    total_loss = 0.0
    dices, ious = [], []

    pbar = tqdm(val_loader, desc='[Eval]', leave=False)
    for inputs, targets, _ in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        dices.append(dice_coeff(logits, targets).item())
        ious.append(iou_coeff(logits, targets).item())

        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         dice=f'{np.mean(dices):.4f}',
                         iou=f'{np.mean(ious):.4f}')

    avg_loss = total_loss / max(1, len(val_loader))
    avg_dice = float(np.mean(dices)) if dices else 0.0
    avg_iou = float(np.mean(ious)) if ious else 0.0
    return avg_loss, avg_dice, avg_iou


def train(args):
    device = get_device()
    print(f'[Device] {device}')

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Directories ----
    save_dir = Path(args.save_dir)
    dgf_flag = f'dgf{args.dgf_ratio}' if args.use_dgf else 'nodgf'
    uag_flag = 'uag' if getattr(args, 'use_uag', False) else 'nouag'
    adapter_flag = getattr(args, 'dgf_adapter', 'mlp')
    llar_flag = 'llar' if getattr(args, 'use_llar', False) else 'nollar'
    l3_flag = f'l3r{getattr(args, "l3_rank", 8)}' if getattr(args, 'use_l3', False) else 'nol3'
    exp_name = f'v1.20_segdino_{args.dino_size}_{args.dataset}_{dgf_flag}_{adapter_flag}_{uag_flag}_{llar_flag}_{l3_flag}_s{args.seed}'
    exp_dir = save_dir / exp_name
    ckpt_dir = exp_dir / 'checkpoints'
    for d in [exp_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f'[Experiment] {exp_name}')
    print(f'[Config] use_dgf={args.use_dgf}, dgf_ratio={args.dgf_ratio}, '
          f'dgf_adapter={getattr(args, "dgf_adapter", "mlp")}, '
          f'use_uag={getattr(args, "use_uag", False)}, '
          f'use_llar={getattr(args, "use_llar", False)}, '
          f'use_l3={getattr(args, "use_l3", False)}, l3_rank={getattr(args, "l3_rank", 8)}, '
          f'seed={args.seed}, epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}')

    # ---- Datasets (split_seed=42 FIXED, independent of training seed) ----
    split_seed = 42
    train_dataset = MedicalSegDataset(
        root=os.path.join(args.data_dir, args.dataset),
        split='train', split_seed=split_seed, img_size=(args.img_size, args.img_size),
        img_dir_name=args.img_dir_name, label_dir_name=args.label_dir_name,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )
    val_dataset = MedicalSegDataset(
        root=os.path.join(args.data_dir, args.dataset),
        split='val', split_seed=split_seed, img_size=(args.img_size, args.img_size),
        img_dir_name=args.img_dir_name, label_dir_name=args.label_dir_name,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)

    print(f'[Split] train={len(train_dataset)}, val={len(val_dataset)} (split_seed={split_seed})')

    # ---- Save split IDs (for reproducibility) ----
    test_dataset = MedicalSegDataset(
        root=os.path.join(args.data_dir, args.dataset),
        split='test', split_seed=split_seed, img_size=(args.img_size, args.img_size),
        img_dir_name=args.img_dir_name, label_dir_name=args.label_dir_name,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )

    import json
    val_ids = [os.path.splitext(os.path.basename(p[0]))[0] for p in val_dataset.pairs]
    test_ids = [os.path.splitext(os.path.basename(p[0]))[0] for p in test_dataset.pairs]
    with open(exp_dir / 'val_ids.json', 'w') as f:
        json.dump(val_ids, f, indent=2)
    with open(exp_dir / 'test_ids.json', 'w') as f:
        json.dump(test_ids, f, indent=2)
    print(f'[Split IDs] val={len(val_ids)}, test={len(test_ids)} → {exp_dir}')

    # ---- Model ----
    backbone = load_backbone(args)
    model = build_model(args, backbone)
    model = model.to(device)

    params = count_all_params(model, backbone)
    print(f'[Params] backbone={params["backbone"]/1e6:.1f}M '
          f'dgf={params["dgf"]/1e6:.3f}M decoder={params["decoder"]/1e6:.3f}M '
          f'trainable={params["total_trainable"]/1e6:.3f}M frozen={params["total_frozen"]/1e6:.1f}M')

    # ---- Optimizer & Loss ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    # ---- CSV Logging ----
    import csv
    log_csv_path = exp_dir / 'log.csv'
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'loss', 'dice', 'iou',
                         'val_loss', 'val_dice', 'val_iou'])
    print(f'[Log] {log_csv_path}')

    # ---- Training loop ----
    best_val_dice = 0.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_dices = []

        pbar = tqdm(train_loader, desc=f'[Train e{epoch}/{args.epochs}]')
        for step, (inputs, targets, _) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dices.append(dice_coeff(logits, targets).item())

            avg_loss = train_loss / (step + 1)
            avg_dice = float(np.mean(train_dices))
            pbar.set_postfix(loss=f'{avg_loss:.4f}', dice=f'{avg_dice:.4f}')

        # ---- Validation ----
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device, criterion)
        train_loss_avg = train_loss / len(train_loader)
        train_dice_avg = float(np.mean(train_dices)) if train_dices else 0.0
        train_iou_avg = float(np.mean([d / (2 - d) for d in train_dices])) if train_dices else 0.0

        print(f'[Epoch {epoch:3d}] train_loss={train_loss_avg:.4f} '
              f'val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}')

        # ---- CSV log ----
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, optimizer.param_groups[0]['lr'],
                             f'{train_loss_avg:.6f}', f'{train_dice_avg:.6f}', f'{train_iou_avg:.6f}',
                             f'{val_loss:.6f}', f'{val_dice:.6f}', f'{val_iou:.6f}'])

        # ---- Checkpoint: best val Dice (SegDINO V2 convention) ----
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            # Remove previous best
            for old in ckpt_dir.glob('best_*.pth'):
                old.unlink()
            best_path = ckpt_dir / f'best_ep{epoch:03d}_dice{val_dice:.4f}_iou{val_iou:.4f}.pth'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, best_path)
            print(f'  [Best] val_dice={val_dice:.4f} @ epoch {epoch}')

    # ---- Summary ----
    print(f'{"="*60}')
    print(f'[Summary] best_val_dice={best_val_dice:.4f} @ epoch {best_epoch}')
    print(f'[Output] {exp_dir}')
    print(f'{"="*60}')

    # ---- Save config.yml ----
    import yaml
    config_dict = {k: v for k, v in sorted(vars(args).items())}
    with open(exp_dir / 'config.yml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    print(f'[Config] {exp_dir / "config.yml"}')

    # ---- Copy best checkpoint as model.pth ----
    best_ckpts = sorted(ckpt_dir.glob('best_*.pth'))
    if best_ckpts:
        import shutil
        model_pth = exp_dir / 'model.pth'
        shutil.copy2(str(best_ckpts[-1]), str(model_pth))
        print(f'[Model] {model_pth}')

    return exp_dir, best_val_dice


def main():
    parser = argparse.ArgumentParser(description='AC-3: SegDINO + DGF Training')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='busi')
    parser.add_argument('--img_dir_name', type=str, default='images')
    parser.add_argument('--label_dir_name', type=str, default='masks')
    parser.add_argument('--img_size', type=int, default=252,
                        help='Image size (252 = 14x18 for DINOv2 patch 14)')
    parser.add_argument('--train_ratio', type=float, default=0.70,
                        help='Fraction for training split (default 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Fraction for validation split (default 0.15)')

    # Model
    parser.add_argument('--dino_version', type=int, default=2,
                        help='2 = DINOv2 (public, auto-download), 3 = DINOv3 (needs --dino_ckpt)')
    parser.add_argument('--dino_repo', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'segdino', 'dinov3-main', 'dinov3-main'))
    parser.add_argument('--dino_ckpt', type=str, default='',
                        help='Path to DINOv3 pretrained weights (.pth) — only for --dino_version 3')
    parser.add_argument('--dino_size', type=str, default='s', choices=['s', 'b'])
    parser.add_argument('--dino_weight', type=str, default='',
                        help='Path to DINOv2 pretrained weights (.pth) — uses local file, no network')
    parser.add_argument('--decoder_dim', type=int, default=256)
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=14,
                        help='ViT patch size (14 for DINOv2 ViT-S/14, 16 for DINOv3 ViT-S/16)')

    # DGF
    parser.add_argument('--use_dgf', type=int, default=1,
                        help='1 = apply DGF adapters, 0 = pure SegDINO baseline')
    parser.add_argument('--dgf_ratio', type=float, default=0.5,
                        help='Fraction of blocks with DGF adapters')
    parser.add_argument('--dgf_adapter', type=str, default='mlp', choices=['mlp', 'dualkernel', 'cavpt', 'pro'],
                        help='mlp = original DGF pointwise prompt, dualkernel = module A dual-kernel conv, '
                             'cavpt = DGF + CAVPT cross-attention prompt')

    # L3 (Lean Linear Layer, dynamic low-rank scale-shift on block linears)
    parser.add_argument('--use_l3', type=int, default=0,
                        help='1 = wrap all block linear layers with LeanLinear (L3Seg)')
    parser.add_argument('--l3_rank', type=int, default=8,
                        help='low-rank dimension for LeanLinear')
    parser.add_argument('--l3_blocks', type=str, default='all', choices=['all', 'odd', 'even', 'head', 'tail'],
                        help='which blocks LeanLinear wraps: all/odd/even/head([0,1])/tail(last two)')
    parser.add_argument('--l3_mode', type=str, default='both', choices=['both', 'attn', 'mlp'],
                        help='which branch LeanLinear wraps per block (attn-only / mlp-only / both)')

    # Stem adapter (Direction AI, X-selection slot B: feature-space prompt at patch_embed)
    parser.add_argument('--use_stem', type=int, default=0,
                        help='1 = wrap patch_embed with a StemAdapter (frozen patch_embed)')

    # LLAR (module B: decoder-side low-level rescue branch)
    parser.add_argument('--use_llar', type=int, default=0,
                        help='1 = fuse raw-image low-level branch into decoder l1')

    # UAG (v1.15)
    parser.add_argument('--use_uag', type=int, default=0,
                        help='1 = use Uncertainty-Adaptive Gate at l4')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42,
                        help='Training seed (42, 1457, 911 for multi-seed)')
    parser.add_argument('--save_dir', type=str, default='./runs')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
