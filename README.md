# DuoAdapt: Dual-Level Orthogonal Adaptation of a Frozen Foundation Model for Multi-Modal Medical Image Segmentation

Reference implementation of the method described in our manuscript (under review). The
framework freezes a DINOv2 ViT-S/14 backbone and combines two lightweight adapters that act
at **orthogonal levels and complementary depths**:

| Adapter | Level | Placement | Role |
|---|---|---|---|
| **DGF** — Domain-adaptive Gate Fusion prompt | token level (pre-block, additive) | 50% of the blocks, interleaved (0, 2, 4, 6, 8, 10) | re-aligns the frozen token stream toward the target clinical domain |
| **L³** — Lean Linear Layer | weight level (dynamic low-rank scale-shift) | the two deepest blocks (10, 11) | per-sample, input-dependent refinement before the decoder consumes the features |

Both adapters are initialized near the identity, so training starts at the frozen baseline and
adapts progressively. The decoder is the Token Pyramid Adaptation + Scale-Aware Decoding
(TPA + SAD) decoder of SegDINO.

Trainable parameters: **4.68 M** (≈17.5% of the 26.74 M total); average Dice **0.8777** over
five public datasets spanning four modalities (single seed 42).

## Repository layout

```
train.py          training entry point (CLI, all paper settings exposed as flags)
test.py           per-image evaluation (Dice / IoU / HD95 / Precision / Recall / Specificity / Accuracy)
dataset.py        dataset with the fixed 70/15/15 split (split seed 42, test set never observed)
dgf_adapter.py    DGF prompt adapters (the paper uses the default `mlp` variant)
dpt.py            DPT-style decoder (TPA + SAD) with optional adapter hooks
utils.py          numpy metric implementations
uag.py            optional auxiliary gate module (unused by the paper's model)
dinov2/           DINOv2 backbone package (ViT-S/14), MIT/Apache-2.0 third-party code
```

## Installation

```bash
pip install -r requirements.txt
# install torch/torchvision matching your CUDA version, e.g.
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

DINOv2 weights are fetched by `dinov2.hub.backbones.dinov2_vits14(pretrained=True)` (downloads
once from the official release) or can be given explicitly with `--dino_weight /path/to/dinov2_vits14_pretrain.pth`
for fully offline runs.

## Data preparation

Each dataset is a directory with paired images and masks (matched by filename stem):

```
data/
  busi/
    images/   *.png
    masks/    *.png
  cvc/
  kvasir/
  isic/
  tn3k/
```

The split (70/15/15, seed 42) is created deterministically by `dataset.py` from the sorted file
list; the test split is never used for checkpoint selection. Please obtain each dataset from its
original publication and follow the corresponding license:

- BUSI — Al-Dhabyani et al., *Data in Brief* 28:104863 (2020)
- CVC-ClinicDB — Bernal et al., *Computerized Medical Imaging and Graphics* 43:99–111 (2015)
- Kvasir-SEG — Jha et al., *MultiMedia Modeling* (2020)
- ISIC 2018 (Task 1) — Codella et al., arXiv:1902.03368
- TN3K — Kang et al., *Medical Image Analysis* 79:102443 (2022)

All images are resized to 252×252 (a multiple of the ViT patch size 14); masks are thresholded
at 0.5 to obtain binary targets.

## Training the four configurations of the ablation

```bash
# G0 — backbone + decoder only
python train.py --dataset busi --use_dgf 0 --use_l3 0 --seed 42

# DGF alone
python train.py --dataset busi --use_dgf 1 --dgf_ratio 0.5 --use_l3 0 --seed 42

# L³ alone
python train.py --dataset busi --use_dgf 0 --use_l3 1 --l3_blocks tail --l3_rank 8 --seed 42

# DuoAdapt (DGF + L³) — the model reported in the paper
python train.py --dataset busi --dgf_ratio 0.5 \
                --use_l3 1 --l3_blocks tail --l3_rank 8 --seed 42
```

Paper settings: 50 epochs, batch size 4, AdamW (lr 1e-4, weight decay 1e-4), BCE-with-logits
loss, best checkpoint selected by validation Dice. Repeat with `--dataset cvc|kvasir|isic|tn3k`.

## Evaluation

```bash
python test.py --dataset busi --ckpt runs/<exp>/checkpoints/best_*.pth
```

Reports per-image Dice, IoU, HD95, Precision, Recall, Specificity and Accuracy, averaged over
the test split (infinite HD95 values from empty predictions are excluded before averaging).

## Reported results (single seed 42)

| Dataset | G0 | DGF alone | L³ alone | **DuoAdapt** |
|---|---|---|---|---|
| BUSI | 0.7117 | 0.8049 | 0.7565 | **0.8133** |
| CVC-ClinicDB | 0.8917 | 0.9191 | 0.9077 | **0.9272** |
| Kvasir-SEG | 0.8558 | 0.9138 | 0.9024 | **0.9263** |
| ISIC 2018 | 0.8937 | **0.9056** | 0.9009 | 0.9053 |
| TN3K | 0.7723 | 0.8084 | 0.7878 | **0.8165** |
| Average | 0.8250 | 0.8704 | 0.8511 | **0.8777** |

DuoAdapt improves over L³ on all five datasets and over DGF on four of the five; the exception is
ISIC 2018, where DuoAdapt and DGF are statistically tied (−0.03 pp) and reported as such.

Please refer to the manuscript tables for the full seven-metric, per-dataset numbers and for the
comparison with six external baselines. Candidate qualitative examples are real model
predictions; aggregate claims are carried by the tables, not by the selected examples.

## Notes on comparability

- Single seed (42) for every method; differences are reported as point estimates, described as
  additive / tied rather than as significance claims.
- The proposed model runs at 252×252, external baselines at their native 256×256. The ≈3%
  pixel-area difference is negligible for method comparison and is stated for transparency.
- FLOPs are counted with `thop`, which does not instrument raw `x @ A @ B` products; the L³ cost
  was added manually (55.0 → 55.2 GFLOPs).

## Citation

```bibtex
@article{duoadapt,
  title   = {DuoAdapt: Dual-Level Orthogonal Adaptation of a Frozen Foundation Model
             for Multi-Modal Medical Image Segmentation},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

The entry will be updated once the manuscript is published.

## License and acknowledgements

Our code is released under the MIT License (see `LICENSE`). It builds on:

- **DINOv2** (`dinov2/`) — Meta AI Research, Apache-2.0; see `THIRD_PARTY_NOTICES.md`.
- **SegDINO** — decoder structure (TPA + SAD) and intermediate-layer selection; please cite the
  SegDINO paper.
- **L³Seg** — the Lean Linear Layer formulation wrapped by our depth-selective placement.
- **UNeXt-pytorch** — repository skeleton (MIT).
