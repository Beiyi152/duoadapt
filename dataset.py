"""
Dataset with strict 70/15/15 train/val/test split.

Split procedure:
1. Collect all (image, mask) pairs
2. Sort by filename for deterministic ordering
3. Split with seed=42: 70% train, 15% val, 15% test
4. Split is FIXED — does not change with training seed
5. Test set is NEVER seen during training or checkpoint selection

Format: root/
  image/  (images, any extension)
  mask/   (masks, matching names, any extension)
"""

import os
import random
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, ColorJitter


SUPPORTED_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _list_files(dir_path: str) -> List[str]:
    out = []
    for f in sorted(os.listdir(dir_path)):
        if not f.startswith('.') and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
            out.append(f)
    return out


def _make_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """Match images to masks by filename stem."""
    images = _list_files(image_dir)
    masks = _list_files(mask_dir)

    image_stems = {os.path.splitext(f)[0]: f for f in images}
    mask_stems = {os.path.splitext(f)[0]: f for f in masks}

    pairs = []
    for stem in sorted(image_stems.keys()):
        if stem in mask_stems:
            pairs.append((
                os.path.join(image_dir, image_stems[stem]),
                os.path.join(mask_dir, mask_stems[stem]),
            ))
        else:
            # Try with extensions ignored (common in medical datasets)
            pass

    if len(pairs) == 0:
        raise RuntimeError(f"No image-mask pairs found in {image_dir} / {mask_dir}")
    return pairs


def split_pairs(pairs: List, seed: int = 42,
                train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[List, List, List]:
    """Split pairs into train/val/test with fixed seed.

    Default: 70/15/15. For KVASIR official split use 80/10/10.
    """
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        [pairs[i] for i in train_idx],
        [pairs[i] for i in val_idx],
        [pairs[i] for i in test_idx],
    )


class TrainTransform:
    """Training augmentations: resize, flip, rotation, color jitter, normalize."""

    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)

    def __call__(self, img, mask):
        img = TF.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=InterpolationMode.NEAREST)

        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            angle = random.randint(-15, 15)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() > 0.5:
            img = self.color_jitter(img)

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return img, mask


class ValTransform:
    """Validation/Test: resize + normalize only, no augmentation."""

    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def __call__(self, img, mask):
        img = TF.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=InterpolationMode.NEAREST)
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return img, mask


class MedicalSegDataset(Dataset):
    """Medical image segmentation dataset with configurable split ratios.

    Args:
        root: Root directory containing 'image/' and 'mask/' subdirectories
        split: 'train', 'val', or 'test'
        split_seed: Seed for split randomization (default 42, FIXED)
        img_size: (H, W) tuple
        train_ratio: fraction of data for training (default 0.70)
        val_ratio: fraction of data for validation (default 0.15)
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        split_seed: int = 42,
        img_size: tuple = (256, 256),
        img_dir_name: str = 'image',
        label_dir_name: str = 'mask',
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir_name = img_dir_name
        self.label_dir_name = label_dir_name

        image_dir = os.path.join(root, img_dir_name)
        mask_dir = os.path.join(root, label_dir_name)

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        all_pairs = _make_pairs(image_dir, mask_dir)
        train_pairs, val_pairs, test_pairs = split_pairs(
            all_pairs, seed=split_seed,
            train_ratio=train_ratio, val_ratio=val_ratio,
        )

        if split == 'train':
            self.pairs = train_pairs
            self.transform = TrainTransform(img_size)
        elif split == 'val':
            self.pairs = val_pairs
            self.transform = ValTransform(img_size)
        elif split == 'test':
            self.pairs = test_pairs
            self.transform = ValTransform(img_size)
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        print(f"[Dataset] {root} | split={split} | {len(self.pairs)} pairs | seed={split_seed}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img_t, mask_t = self.transform(img, mask)
        mask_t = (mask_t > 0.5).float()

        name = os.path.splitext(os.path.basename(img_path))[0]
        return img_t, mask_t, name
