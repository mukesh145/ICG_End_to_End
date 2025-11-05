# icg/models/encoder.py
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models

class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B0 feature extractor.
    Input:  [B,3,224,224]
    Output: [B,C,H,W] with C=1280 and spatial ~7x7 (for 224 crop)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Torchvision 0.17+: weights arg; fallback for older versions if needed
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
        except AttributeError:
            # Older API
            backbone = models.efficientnet_b0(pretrained=pretrained)

        # Keep only the convolutional feature extractor
        self.features = backbone.features
        # No classifier / pooling; we use spatial map
        self.out_channels = 1280  # EfficientNet-B0 final feature dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] -> feats: [B,1280,H',W']
        """
        feats = self.features(x)
        return feats
