"""
DPT model with optional DGF adapter support.

Supports both DINOv2 (public) and DINOv3 (gated) backbones.
Based on SegDINO V2 TPASADDecoder architecture.
When use_dgf=True, the backbone is wrapped with DGF adapters at 50% of blocks.

Modified from: segdino_v2/dpt.py (MICCAI 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgf_adapter import apply_dgf_to_backbone, set_backbone_prompt
from uag import UncertaintyAdaptiveGate


class ResidualDepthwiseBlock(nn.Module):
    """Residual depthwise separable convolution block with learnable gamma."""

    def __init__(self, channels, use_group_norm=True):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = (
            nn.GroupNorm(min(32, channels), channels)
            if use_group_norm
            else nn.BatchNorm2d(channels)
        )
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = self.act(self.norm(self.pointwise(self.depthwise(x))))
        return x + self.gamma * residual


class TPAResampleProject(nn.Module):
    """Token Pyramid Alignment: resample + project a feature branch."""

    def __init__(self, channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return self.conv(x)


class LowLevelRescue(nn.Module):
    """Decoder-side low-level rescue branch (module B).

    Injects raw-image intensity/texture into the finest decoder scale (l1),
    giving the frozen ViT a direct low-level pathway it otherwise lacks
    (TPASAD only consumes intermediate ViT features [2,5,8,11]).

    Borrowed from Dino U-Net SPM / U-DFA SPA, but fused into the DECODER
    (not the encoder), which is our differentiation point.

    Fail-safe: gamma init=0 → training starts exactly at the no-LLAR baseline.
    """

    def __init__(self, in_channels=3, channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, 64), 64),
            nn.GELU(),
            nn.Conv2d(64, channels, 3, padding=1, bias=False),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, raw, out_h, out_w):
        x = self.stem(raw)
        if x.shape[-2:] != (out_h, out_w):
            x = F.interpolate(x, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return self.gamma * x


class TPASADDecoder(nn.Module):
    """Token Pyramid Alignment + Scale-Aware Decoder (SegDINO V2).

    When use_uag=True, an Uncertainty-Adaptive Gate is applied at l4
    (deepest bottleneck position): variance-gated residual refinement.
    """

    def __init__(self, in_dims, decoder_channels=256, num_classes=1, use_group_norm=True,
                 use_uag=False, use_llar=False):
        super().__init__()
        assert len(in_dims) == 4

        self.token_projections = nn.ModuleList(
            [nn.Conv2d(ch, decoder_channels, 1, bias=False) for ch in in_dims]
        )
        self.tpa_branch_1 = TPAResampleProject(decoder_channels, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProject(decoder_channels, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProject(decoder_channels, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProject(decoder_channels, scale_factor=1)

        self.sad_intra_1 = ResidualDepthwiseBlock(decoder_channels, use_group_norm)
        self.sad_intra_2 = ResidualDepthwiseBlock(decoder_channels, use_group_norm)
        self.sad_intra_3 = ResidualDepthwiseBlock(decoder_channels, use_group_norm)
        self.sad_intra_4 = ResidualDepthwiseBlock(decoder_channels, use_group_norm)

        self.sad_inter_4 = ResidualDepthwiseBlock(decoder_channels, use_group_norm)
        self.sad_inter_3 = ResidualDepthwiseBlock(decoder_channels, use_group_norm)
        self.sad_inter_2 = ResidualDepthwiseBlock(decoder_channels, use_group_norm)
        self.sad_inter_1 = ResidualDepthwiseBlock(decoder_channels, use_group_norm)

        self.out_conv = nn.Conv2d(decoder_channels, num_classes, 1)

        self.use_uag = use_uag
        if use_uag:
            self.uag_l4 = UncertaintyAdaptiveGate(decoder_channels)

        self.use_llar = use_llar
        if use_llar:
            self.llar = LowLevelRescue(in_channels=3, channels=decoder_channels)

    def _tokens_to_feature_map(self, x, patch_h, patch_w):
        if isinstance(x, (list, tuple)):
            x = x[0]
        num_patches = patch_h * patch_w
        if x.ndim != 3:
            raise ValueError(f"Expected 3D token tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != num_patches:
            x = x[:, -num_patches:, :]
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

    def forward(self, features, patch_h, patch_w, raw=None):
        branches = []
        for idx, tokens in enumerate(features):
            fm = self._tokens_to_feature_map(tokens, patch_h, patch_w)
            fm = self.token_projections[idx](fm)
            branches.append(fm)

        b1 = self.tpa_branch_1(branches[0])
        b2 = self.tpa_branch_2(branches[1])
        b3 = self.tpa_branch_3(branches[2])
        b4 = self.tpa_branch_4(branches[3])

        l1 = self.sad_intra_1(b1)
        l2 = self.sad_intra_2(b2)
        l3 = self.sad_intra_3(b3)
        l4 = self.sad_intra_4(b4)

        if self.use_llar and raw is not None:
            l1 = l1 + self.llar(raw, l1.shape[-2], l1.shape[-1])

        if self.use_uag:
            l4 = self.uag_l4(l4)

        x4 = self.sad_inter_4(l4)
        x3_up = F.interpolate(x4, size=l3.shape[-2:], mode='bilinear', align_corners=False)
        x3 = self.sad_inter_3(x3_up + l3)

        x2_up = F.interpolate(x3, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        x2 = self.sad_inter_2(x2_up + l2)

        x1_up = F.interpolate(x2, size=l1.shape[-2:], mode='bilinear', align_corners=False)
        x1 = self.sad_inter_1(x1_up + l1)

        return self.out_conv(x1)


class DPT(nn.Module):
    """DPT model with frozen DINOv2/DINOv3 backbone + TPASAD decoder.

    Args:
        encoder_size: 'small' (ViT-S) or 'base' (ViT-B)
        nclass: Number of output classes (1 for binary segmentation)
        decoder_channels: TPASAD decoder channel width
        patch_size: ViT patch size (14 for DINOv2, 16 for DINOv3)
        use_bn: Use BatchNorm instead of GroupNorm in decoder
        backbone: Pre-loaded DINOv2 or DINOv3 ViT model
        use_dgf: Apply DGF adapters to backbone
        dgf_ratio: Fraction of blocks with DGF (default 0.5)
        dino_version: 2 for DINOv2, 3 for DINOv3 (controls adapter variant)
    """

    def __init__(
        self,
        encoder_size='small',
        nclass=1,
        decoder_channels=256,
        patch_size=16,
        use_bn=False,
        backbone=None,
        use_dgf=False,
        dgf_ratio=0.5,
        dino_version=3,
        use_uag=False,
        use_llar=False,
        dgf_adapter='mlp',
        use_l3=False,
        l3_rank=8,
        use_stem=False,
        l3_blocks='all',
        l3_mode='both',
    ):
        super().__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
        }

        self.encoder_size = encoder_size
        self.patch_size = patch_size
        self.use_dgf = use_dgf
        self.nclass = nclass

        # Apply DGF adapters / L3 / stem adapter before wrapping (if enabled)
        if backbone is not None and (use_dgf or use_l3 or use_stem):
            backbone = apply_dgf_to_backbone(backbone,
                                             dgf_ratio=dgf_ratio if use_dgf else 0,
                                             dino_version=dino_version,
                                             adapter_type=dgf_adapter,
                                             use_l3=use_l3,
                                             l3_rank=l3_rank,
                                             use_stem=use_stem,
                                             l3_blocks=l3_blocks,
                                             l3_mode=l3_mode)

        self.backbone = backbone
        self.in_dims = [self.backbone.embed_dim] * 4
        self.decoder = TPASADDecoder(
            self.in_dims,
            decoder_channels=decoder_channels,
            num_classes=nclass,
            use_group_norm=not use_bn,
            use_uag=use_uag,
            use_llar=use_llar,
        )

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def set_prompt(self, enabled: bool):
        """Toggle DGF prompt for all wrapped blocks (inference-time floor)."""
        set_backbone_prompt(self.backbone, enabled)

    def forward(self, x, return_feats=False):
        patch_h = x.shape[-2] // self.patch_size
        patch_w = x.shape[-1] // self.patch_size
        feats = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size], norm=True
        )
        out = self.decoder(feats, patch_h, patch_w, raw=x)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if return_feats:
            return out, feats[-1]
        return out
