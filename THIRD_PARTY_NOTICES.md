# Third-party notices

This repository contains or derives from the following third-party projects. Their licenses
apply to the corresponding files/portions.

## DINOv2 (`dinov2/`)

Source: Meta AI Research — https://github.com/facebookresearch/dinov2

```
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

The DINOv2 ViT-S/14 weights used as the frozen backbone are distributed by Meta AI Research
under the same license.

## SegDINO

`dpt.py` follows the decoder design (Token Pyramid Adaptation + Scale-Aware Decoding) and the
intermediate-layer selection of SegDINO (MICCAI 2026). Please cite the original paper.

## L³Seg

The Lean Linear Layer used inside `dgf_adapter.py` follows the formulation of L³Seg
(ICCV Workshops / CVAMD 2025). Please cite the original paper.

## UNeXt-pytorch

The training/evaluation skeleton derives from the MIT-licensed UNeXt-pytorch repository
(Copyright (c) 2022 Jeya Maria Jose).
