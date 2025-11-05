# icg/models/model.py
from __future__ import annotations
import torch
import torch.nn as nn

from icg.models.encoder import EfficientNetEncoder
from icg.models.projector import CNNProjector
from icg.models.positional_encoding import SinusoidalPositionalEncoding
from icg.models.decoder import TransformerCaptionDecoder

class CaptioningModel(nn.Module):
    """
    Full image captioning model:
      images -> EfficientNet features -> seq projection -> Transformer decoder -> logits
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        tie_weights: bool = True,
        encoder_pretrained: bool = True,
    ):
        super().__init__()
        # Encoder
        self.encoder = EfficientNetEncoder(pretrained=encoder_pretrained)
        # Projector (C->d_model, flatten HxW)
        self.projector = CNNProjector(in_channels=self.encoder.out_channels, d_model=d_model, dropout=dropout)
        # Positional encodings for src (S) and tgt (T)
        self.src_pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=5000, dropout=dropout)
        self.tgt_pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=500, dropout=dropout)

        # Decoder stack
        self.decoder = TransformerCaptionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            tie_weights=tie_weights,
        )

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,3,224,224]
        returns memory: [S,B,d_model]
        """
        feats = self.encoder(images)          # [B,C,H',W']
        src_seq = self.projector(feats)       # [S,B,d_model]
        return src_seq

    def forward(
        self,
        images: torch.Tensor,                      # [B,3,H,W]
        tgt_in: torch.Tensor,                      # [B,T] -> will be transposed to [T,B]
        tgt_attn_mask: torch.Tensor | None = None, # [T,T]
        tgt_key_padding_mask: torch.Tensor | None = None,  # [B,T]
    ) -> torch.Tensor:
        """
        Returns logits: [B,T,V]
        """
        # Encode
        memory = self.encode_images(images)   # [S,B,E]

        # Transformer expects [T,B]; collate gave [B,T]
        tgt_in_TB = tgt_in.transpose(0, 1)    # [T,B]

        # Forward through decoder
        logits_TBV = self.decoder(
            tgt_in=tgt_in_TB,
            memory=memory,
            tgt_pos_enc=self.tgt_pos_enc,
            src_pos_enc=self.src_pos_enc,
            tgt_attn_mask=tgt_attn_mask,                # already [T,T]
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,T]
            memory_key_padding_mask=None,
        )  # [T,B,V]

        logits = logits_TBV.transpose(0, 1).contiguous()  # [B,T,V]
        return logits
