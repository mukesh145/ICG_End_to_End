# icg/models/projector.py
from __future__ import annotations
import torch
import torch.nn as nn

class CNNProjector(nn.Module):
    """
    Flattens spatial features [B,C,H,W] -> sequence [S,B,d_model]
    and linearly projects channel dim C -> d_model.
    """
    def __init__(self, in_channels: int = 1280, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.proj = nn.Linear(in_channels, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B,C,H,W]
        returns: src_seq [S,B,d_model], where S=H*W
        """
        B, C, H, W = feats.shape
        x = feats.permute(0, 2, 3, 1).contiguous()     # [B,H,W,C]
        x = x.view(B, H * W, C)                        # [B,S,C]
        x = self.proj(x)                               # [B,S,d_model]
        x = self.dropout(x)
        x = x.permute(1, 0, 2).contiguous()            # [S,B,d_model] for nn.Transformer
        return x
