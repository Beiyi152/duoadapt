"""
DGF (Domain-Gap Fine-tuning) Adapter for frozen ViT backbones.

Exact pattern from SGNNet (BSPC 2026): pre-block prompt adapter.
Reference: github.com/platgs2013/SGNNet — Adapter_linear class.

Supports both DINOv2 (no rope) and DINOv3 (rope) block interfaces.

Design:
- Bottleneck MLP: Linear(dim→256) → GELU → Linear(256→dim) → GELU
- Applied as a learned PROMPT (added to input BEFORE the ViT block)
- Applied to 50% of blocks (every other, starting from block 0)
- Wraps the original frozen ViT block; prompt_learn is trainable, block is frozen

v1.20 additions (prompt-combination):
- PromptCAVPTAdapter*: DGF pointwise prompt + CAVPT cross-attention prompt
  (learnable prompt tokens as query, DVPT / Neural Networks 2025).
- LeanLinear (L3): dynamic low-rank scale-shift applied to the frozen block's
  linear layers (qkv/proj/w12/w3), from L3Seg (ICCV 2025 Workshop).
"""

import math

import torch
import torch.nn as nn


# Submodule names that are trainable inside an adapter (everything else is frozen).
_ADAPTER_TRAINABLE_KEYS = ('prompt_learn', 'conv_large', 'conv_small', 'bottleneck', 'hyper')


def is_adapter_param(name: str) -> bool:
    """True if `name` belongs to a trainable adapter submodule."""
    return any(k in name for k in _ADAPTER_TRAINABLE_KEYS)


class DGFAdapterV2(nn.Module):
    """Pre-block prompt adapter for DINOv2 blocks (no rope argument).

    Pattern: x = x + prompt_learn(x); x = block(x)
    """

    def __init__(self, blk: nn.Module, dim: int):
        super().__init__()
        self.block = blk
        self.use_prompt = True
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, dim),
            nn.GELU(),
        )

    def forward(self, x):
        prompted = x + self.prompt_learn(x) if self.use_prompt else x
        return self.block(prompted)


class DGFAdapter(nn.Module):
    """Pre-block prompt adapter for DINOv3 blocks (with rope argument).

    Pattern: x = x + prompt_learn(x); x = block(x, rope)
    """

    def __init__(self, blk: nn.Module, dim: int):
        super().__init__()
        self.block = blk
        self.use_prompt = True
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, dim),
            nn.GELU(),
        )

    def forward(self, x_or_list, rope_or_rope_list=None):
        if isinstance(x_or_list, torch.Tensor):
            prompted = x_or_list + self.prompt_learn(x_or_list) if self.use_prompt else x_or_list
        else:
            prompted = [x + self.prompt_learn(x) for x in x_or_list] if self.use_prompt else x_or_list
        return self.block(prompted, rope_or_rope_list)


class DualKernelAdapterV2(nn.Module):
    """Pre-block dual-kernel conv prompt adapter for DINOv2 blocks (module A).

    Unlike the plain DGF MLP adapter (pointwise, no spatial context), this
    adapter reshapes the spatial tokens back to a 2D grid and applies a
    large-kernel + small-kernel depthwise conv BEFORE the pointwise
    bottleneck. This gives the frozen ViT explicit spatial receptive field,
    targeting the diagnosed "low-level appearance non-firing" weakness
    (bright/small lesions).

    Pattern: spatial = spatial + bottleneck(large_conv(grid) + small_conv(grid));
             x = block([CLS, spatial])
    """

    def __init__(self, blk: nn.Module, dim: int):
        super().__init__()
        self.block = blk
        self.use_prompt = True
        self.conv_large = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.conv_small = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, dim),
            nn.GELU(),
        )

    def forward(self, x):
        if not self.use_prompt:
            return self.block(x)
        cls = x[:, :1]
        spatial = x[:, 1:]  # (B, N, C)
        B, N, C = spatial.shape
        h = w = int(math.isqrt(N))
        s2d = spatial.transpose(1, 2).reshape(B, C, h, w)
        f = self.conv_large(s2d) + self.conv_small(s2d)
        f = f.reshape(B, C, N).transpose(1, 2)
        f = self.bottleneck(f)
        prompted = torch.cat([cls, spatial + f], dim=1)
        return self.block(prompted)


class DualKernelAdapter(nn.Module):
    """Dual-kernel conv prompt adapter for DINOv3 blocks (with rope).

    Same spatial conv prompt as DualKernelAdapterV2, but forwards through the
    DINOv3 block interface (x_or_list, rope_or_rope_list).
    """

    def __init__(self, blk: nn.Module, dim: int):
        super().__init__()
        self.block = blk
        self.use_prompt = True
        self.conv_large = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.conv_small = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, dim),
            nn.GELU(),
        )

    def _prompt(self, x):
        cls = x[:, :1]
        spatial = x[:, 1:]
        B, N, C = spatial.shape
        h = w = int(math.isqrt(N))
        s2d = spatial.transpose(1, 2).reshape(B, C, h, w)
        f = self.conv_large(s2d) + self.conv_small(s2d)
        f = f.reshape(B, C, N).transpose(1, 2)
        f = self.bottleneck(f)
        return torch.cat([cls, spatial + f], dim=1)

    def forward(self, x_or_list, rope_or_rope_list=None):
        if not self.use_prompt:
            return self.block(x_or_list, rope_or_rope_list)
        if isinstance(x_or_list, torch.Tensor):
            prompted = self._prompt(x_or_list)
        else:
            prompted = [self._prompt(x) for x in x_or_list]
        return self.block(prompted, rope_or_rope_list)


class PromptCAVPTAdapterV2(nn.Module):
    """Pre-block prompt adapter = DGF pointwise prompt + CAVPT cross-attention.

    CAVPT (DVPT, Neural Networks 2025): a small set of learnable prompt tokens
    acts as QUERY over the (patched) token sequence, producing a cross-attended
    prompt that is injected (gated, init=0) into the first `num_prompt` token
    positions. This realizes a token-interaction mechanism the plain DGF
    pointwise prompt cannot (DGF never lets spatial tokens exchange info).

    Pattern: x = x + prompt_learn(x); x = x + gate * pad(cross_attn(prompt, x));
             x = block(x)
    """

    def __init__(self, blk: nn.Module, dim: int, num_prompt: int = 10):
        super().__init__()
        self.block = blk
        self.use_prompt = True
        # DGF pointwise prompt (kept — this is the DGF+CAVPT combination)
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, dim),
            nn.GELU(),
        )
        # CAVPT cross-attention prompt
        self.num_prompt = num_prompt
        self.prompt = nn.Parameter(torch.zeros(1, num_prompt, dim))
        nn.init.trunc_normal_(self.prompt, std=0.02)
        self.scale = dim ** -0.5
        self.prompt_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if not self.use_prompt:
            return self.block(x)
        prompted = x + self.prompt_learn(x)
        B, N, C = prompted.shape
        # (1,num,C) @ (B,C,N) -> (B,num,N) attention, then (B,num,C) attended output
        attn = torch.softmax(self.prompt @ prompted.transpose(-2, -1) * self.scale, dim=-1)
        prompt_out = attn @ prompted
        pad = torch.zeros(B, N - self.num_prompt, C, device=prompted.device, dtype=prompted.dtype)
        injection = torch.cat([prompt_out, pad], dim=1) * self.prompt_gate
        return self.block(prompted + injection)


class PromptCAVPTAdapter(nn.Module):
    """DGF + CAVPT prompt adapter for DINOv3 blocks (with rope)."""

    def __init__(self, blk: nn.Module, dim: int, num_prompt: int = 10):
        super().__init__()
        self.block = blk
        self.use_prompt = True
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, dim),
            nn.GELU(),
        )
        self.num_prompt = num_prompt
        self.prompt = nn.Parameter(torch.zeros(1, num_prompt, dim))
        nn.init.trunc_normal_(self.prompt, std=0.02)
        self.scale = dim ** -0.5
        self.prompt_gate = nn.Parameter(torch.zeros(1))

    def _prompt(self, x):
        prompted = x + self.prompt_learn(x)
        B, N, C = prompted.shape
        attn = torch.softmax(self.prompt @ prompted.transpose(-2, -1) * self.scale, dim=-1)
        prompt_out = attn @ prompted
        pad = torch.zeros(B, N - self.num_prompt, C, device=prompted.device, dtype=prompted.dtype)
        injection = torch.cat([prompt_out, pad], dim=1) * self.prompt_gate
        return prompted + injection

    def forward(self, x_or_list, rope_or_rope_list=None):
        if not self.use_prompt:
            return self.block(x_or_list, rope_or_rope_list)
        if isinstance(x_or_list, torch.Tensor):
            prompted = self._prompt(x_or_list)
        else:
            prompted = [self._prompt(x) for x in x_or_list]
        return self.block(prompted, rope_or_rope_list)


class LeanLinear(nn.Module):
    """L3 (Lean Linear Layer) dynamic low-rank scale-shift.

    From L3Seg (ICCV 2025 Workshop, Bhardwaj et al.), Eq. 3-5:

        y = (1 + gamma(x)) * (x W0^T + b0) + beta(x)
        gamma(x) = x Ag Bg   (rank-r)
        beta(x)  = x Ab Bb   (rank-r)

    Init: Ag/Ab ~ N(0, 1e-3), Bg/Bb = 0, base frozen.
    This is a fail-safe dynamic low-rank adaptation of the frozen ViT's own
    linear projections (qkv/proj/w12/w3) — orthogonal to the pre-block DGF
    prompt (which touches tokens, not the block's weights).
    """

    def __init__(self, base: nn.Linear, rank: int = 8):
        super().__init__()
        self.base = base
        self.rank = rank
        in_dim, out_dim = base.in_features, base.out_features
        self.Ag = nn.Parameter(torch.randn(in_dim, rank) * 1e-3)
        self.Bg = nn.Parameter(torch.zeros(rank, out_dim))
        self.Ab = nn.Parameter(torch.randn(in_dim, rank) * 1e-3)
        self.Bb = nn.Parameter(torch.zeros(rank, out_dim))
        base.weight.requires_grad = False
        if base.bias is not None:
            base.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        gamma = x @ self.Ag @ self.Bg
        beta = x @ self.Ab @ self.Bb
        return (1.0 + gamma) * base_out + beta


def wrap_block_l3(blk: nn.Module, rank: int = 8, mode: str = 'both') -> nn.Module:
    """Replace a ViT block's linear layers with LeanLinear (in-place).

    Handles DINOv2 attention (qkv/proj) + SwiGLUFFN (w12/w3), with a fallback
    to the standard Mlp (fc1/fc2) interface. `mode` selects which branch is
    wrapped: 'both' (default), 'attn' (attention projections only), or 'mlp'
    (MLP projections only) — the branch axis of L3 instantiation.
    """
    if mode in ('both', 'attn'):
        attn = getattr(blk, 'attn', None)
        if attn is not None:
            for name in ('qkv', 'proj'):
                lin = getattr(attn, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(attn, name, LeanLinear(lin, rank))
    if mode in ('both', 'mlp'):
        mlp = getattr(blk, 'mlp', None)
        if mlp is not None:
            for name in ('w12', 'w3', 'fc1', 'fc2'):
                lin = getattr(mlp, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(mlp, name, LeanLinear(lin, rank))
    return blk


class DGFProAdapterV2(nn.Module):
    """DGF-Pro V2: DGF pointwise prompt + sample-adaptive FiLM (Direction AI).

    Stays on the DGF floor: prompt_learn is the original pointwise MLP, and a
    tiny hyper-net reads the pooled image descriptor (mean over spatial tokens)
    and emits per-channel (gamma, beta) that modulate the prompt. gamma=0/beta=0
    at init -> the module IS pure DGF at start (identity floor); the (1+gamma)
    factor acts as a learned gate with nonzero gradient from epoch 1, so the
    model can specialize the prompt per image WITHOUT domain labels.

    Pattern: p = prompt_learn(x); prompted = x + p*(1+gamma) + beta; x = block(x)
    """

    def __init__(self, blk: nn.Module, dim: int):
        super().__init__()
        self.block = blk
        self.use_prompt = True
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, dim),
            nn.GELU(),
        )
        self.hyper = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * dim),
        )
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

    def forward(self, x):
        if not self.use_prompt:
            return self.block(x)
        p = self.prompt_learn(x)
        g = x[:, 1:].mean(dim=1, keepdim=True)   # (B,1,C) image-level descriptor
        out = self.hyper(g)                       # (B,1,2C)
        gamma, beta = out[..., :x.shape[-1]], out[..., x.shape[-1]:]
        prompted = x + p * (1.0 + gamma) + beta
        return self.block(prompted)


class DGFProAdapter(nn.Module):
    """DGF-Pro V2 for DINOv3 blocks (with rope). Same FiLM prompt as V2."""

    def __init__(self, blk: nn.Module, dim: int):
        super().__init__()
        self.block = blk
        self.use_prompt = True
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, dim),
            nn.GELU(),
        )
        self.hyper = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * dim),
        )
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

    def _prompt(self, x):
        p = self.prompt_learn(x)
        g = x[:, 1:].mean(dim=1, keepdim=True)
        out = self.hyper(g)
        gamma, beta = out[..., :x.shape[-1]], out[..., x.shape[-1]:]
        return x + p * (1.0 + gamma) + beta

    def forward(self, x_or_list, rope_or_rope_list=None):
        if not self.use_prompt:
            return self.block(x_or_list, rope_or_rope_list)
        if isinstance(x_or_list, torch.Tensor):
            prompted = self._prompt(x_or_list)
        else:
            prompted = [self._prompt(xt) for xt in x_or_list]
        return self.block(prompted, rope_or_rope_list)


class StemAdapter(nn.Module):
    """Feature-space prompt applied right after the ViT patch embedding (stem).

    Direction AI, X-selection slot B. The pre-block DGF prompt acts per-block on
    the token sequence; this instead acts ONCE, immediately after patch_embed
    (before CLS/pos-encoding), so a single trainable transform conditions all
    downstream layers. Same pointwise-MLP prompt pattern as DGF (hidden=1536 ~
    DGF's ~1.18M budget), but a DIFFERENT slot. inner (patch_embed) stays frozen.
    """

    def __init__(self, patch_embed: nn.Module, dim: int, hidden: int = 1536):
        super().__init__()
        self.inner = patch_embed
        for p in self.inner.parameters():
            p.requires_grad = False
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.GELU(),
        )

    def forward(self, x):
        tokens = self.inner(x)                 # (B, N_patch, C)
        return tokens + self.prompt_learn(tokens)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # delegate attribute access (num_patches, grid_size, ...) to inner
            return getattr(object.__getattribute__(self, 'inner'), name)


def apply_stem_to_backbone(backbone, dim: int) -> nn.Module:
    """Wrap backbone.patch_embed with a StemAdapter (in-place)."""
    backbone.patch_embed = StemAdapter(backbone.patch_embed, dim)
    print(f"[Stem] Applied StemAdapter on patch_embed (dim={dim})")
    return backbone


_PROMPT_ADAPTER_TYPES = (
    DGFAdapter, DGFAdapterV2,
    DualKernelAdapter, DualKernelAdapterV2,
    PromptCAVPTAdapter, PromptCAVPTAdapterV2,
    DGFProAdapter, DGFProAdapterV2,
)


def _set_trainable(backbone):
    """Freeze everything, then unfreeze adapter prompt params + LeanLinear A/B.

    Robust module-based trainability: the wrapped `.block` subtree stays frozen
    (except for any LeanLinear inside it), while adapter prompt modules and
    LeanLinear low-rank factors become trainable.
    """
    for p in backbone.parameters():
        p.requires_grad = False

    for module in backbone.modules():
        if isinstance(module, LeanLinear):
            for p in (module.Ag, module.Bg, module.Ab, module.Bb):
                p.requires_grad = True
        elif isinstance(module, StemAdapter):
            for p in module.prompt_learn.parameters():
                p.requires_grad = True
        elif isinstance(module, _PROMPT_ADAPTER_TYPES):
            for child_name, child in module.named_children():
                if child_name == 'block':
                    continue
                for p in child.parameters():
                    p.requires_grad = True
            for p in module.parameters(recurse=False):
                p.requires_grad = True


def apply_dgf_to_backbone(backbone, dgf_ratio: float = 0.5, dino_version: int = 3,
                          adapter_type: str = 'mlp', use_l3: bool = False,
                          l3_rank: int = 8, use_stem: bool = False,
                          l3_blocks: str = 'all', l3_mode: str = 'both'):
    """Apply DGF/CAVPT prompt adapters + optional L3 + optional stem adapter.

    Args:
        backbone: DINOv2 or DINOv3 ViT model instance
        dgf_ratio: Fraction of blocks to apply the prompt adapter (default 0.5);
            <=0 skips the pre-block prompt entirely (L3/stem-only runs)
        dino_version: 2 for DINOv2, 3 for DINOv3
        adapter_type: 'mlp' (original DGF pointwise prompt), 'cavpt'
            (DGF + CAVPT cross-attention prompt), 'dualkernel' (module A),
            or 'pro' (DGF-Pro)
        use_l3: if True, wrap block linear layers with LeanLinear
        l3_rank: low-rank dimension for LeanLinear (default 8)
        use_stem: if True, wrap patch_embed with a StemAdapter (slot-B probe)
        l3_blocks: which blocks LeanLinear wraps: 'all' | 'odd' | 'even' |
            'head' ([0,1]) | 'tail' (last two) | comma list e.g. '10,11'.
            odd/even make L3 disjoint from DGF (which sits on even blocks).
        l3_mode: which branch L3 wraps per block: 'both' | 'attn' | 'mlp'.

    Returns:
        backbone with prompt-wrapped and/or L3-wrapped blocks (modified in-place)
    """
    dim = backbone.embed_dim
    num_blocks = len(backbone.blocks)

    # 1) optional L3 on selected blocks (BEFORE prompt-adapter wrapping so L3
    #    lives on the raw blocks; _set_trainable keeps its base frozen + A/B trainable)
    if use_l3:
        if l3_blocks == 'all':
            idxs = list(range(num_blocks))
        elif l3_blocks == 'odd':
            idxs = [i for i in range(num_blocks) if i % 2 == 1]
        elif l3_blocks == 'even':
            idxs = [i for i in range(num_blocks) if i % 2 == 0]
        elif l3_blocks == 'head':
            idxs = [0, 1]
        elif l3_blocks == 'tail':
            idxs = [num_blocks - 2, num_blocks - 1]
        else:
            idxs = [int(i) for i in l3_blocks.split(',')]
        for i in idxs:
            backbone.blocks[i] = wrap_block_l3(backbone.blocks[i], l3_rank, l3_mode)
        print(f"[L3] Applied LeanLinear (rank={l3_rank}, l3_blocks={l3_blocks}, "
              f"l3_mode={l3_mode}) to {len(idxs)}/{num_blocks} blocks: {idxs}")

    # 2) optional stem adapter (feature-space prompt right after patch_embed)
    if use_stem:
        apply_stem_to_backbone(backbone, dim)

    # 3) prompt adapters on selected blocks (skip when dgf_ratio <= 0)
    if dgf_ratio > 0:
        num_dgf = max(int(num_blocks * dgf_ratio), 1)
        step = num_blocks / num_dgf
        layer_indices = [int(i * step) for i in range(num_dgf)]

        if adapter_type == 'cavpt':
            AdapterCls = PromptCAVPTAdapterV2 if dino_version == 2 else PromptCAVPTAdapter
        elif adapter_type == 'pro':
            AdapterCls = DGFProAdapterV2 if dino_version == 2 else DGFProAdapter
        elif adapter_type == 'dualkernel':
            AdapterCls = DualKernelAdapterV2 if dino_version == 2 else DualKernelAdapter
        else:
            AdapterCls = DGFAdapterV2 if dino_version == 2 else DGFAdapter
        print(f"[DGF] Applying {AdapterCls.__name__} to {len(layer_indices)}/{num_blocks} blocks: {layer_indices}")

        for idx in layer_indices:
            backbone.blocks[idx] = AdapterCls(backbone.blocks[idx], dim=dim)

    _set_trainable(backbone)
    return backbone


def set_backbone_prompt(backbone, enabled: bool):
    """Toggle the DGF/CAVPT prompt on all wrapped blocks (inference-time floor).

    When enabled=False, adapted blocks skip the prompt and behave as the
    frozen ViT, giving a prompt-off forward used by the no-harm safety floor.
    """
    for blk in backbone.blocks:
        if hasattr(blk, 'use_prompt'):
            blk.use_prompt = enabled


def count_params(backbone) -> dict:
    """Count backbone parameters by trainability."""
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    return {
        'trainable': trainable,
        'frozen': frozen,
        'total': trainable + frozen,
        'trainable_pct': 100 * trainable / (trainable + frozen) if (trainable + frozen) > 0 else 0,
    }
