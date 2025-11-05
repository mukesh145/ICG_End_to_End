# icg/models/decoder.py
from __future__ import annotations
import torch
import torch.nn as nn

class TransformerCaptionDecoder(nn.Module):
    """
    Transformer decoder stack for captioning.
    - Token embedding + positional encoding are provided externally (wrapper).
    - Uses nn.TransformerDecoder under the hood.
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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # we operate [T,B,E]
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.out_proj.weight = self.tok_emb.weight  # weight tying

    def forward(
        self,
        tgt_in: torch.Tensor,                     # [T,B] token ids (BOS ... )
        memory: torch.Tensor,                     # [S,B,d_model] encoder seq
        tgt_pos_enc,                              # callable: x->[T,B,E]
        src_pos_enc,                              # callable: x->[S,B,E]
        tgt_attn_mask: torch.Tensor | None = None,         # [T,T] causal float mask (-inf/0)
        tgt_key_padding_mask: torch.Tensor | None = None,  # [B,T] True where PAD
        memory_key_padding_mask: torch.Tensor | None = None, # (unused; src has no pad)
    ) -> torch.Tensor:
        """
        Returns logits: [T,B,V]
        """
        # Embed targets
        x = self.tok_emb(tgt_in)                  # [T,B,E]
        x = tgt_pos_enc(x)                        # add positional enc

        mem = src_pos_enc(memory)                 # position on encoder seq

        # nn.Transformer expects masks in specific dtypes/shapes; we pass as-is.
        h = self.decoder(
            tgt=x,
            memory=mem,
            tgt_mask=tgt_attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )                                         # [T,B,E]

        logits = self.out_proj(h)                 # [T,B,V]
        return logits
