# icg/data/collate.py
from __future__ import annotations
import torch
from typing import List, Dict

def _pad_sequences(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    """
    Pad a list of variable-length int sequences into a LongTensor [B, T]
    """
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), fill_value=pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out

def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Causal mask for decoding: [T,T] with True/1 at masked places.
    We'll follow PyTorch Transformer convention (float mask with -inf).
    """
    # Upper triangular (including diagonal above) should be masked.
    mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)
    # Convert to float mask with -inf where masked, 0 elsewhere
    float_mask = mask.masked_fill(mask, float('-inf')).float()
    return float_mask  # [T,T]

def _make_padding_mask(batch_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Returns [B, T] bool mask where True indicates PAD positions.
    """
    return (batch_ids == pad_id)

class CaptionCollate:
    """
    Produces:
      - images: FloatTensor [B,3,H,W]
      - tgt_in: LongTensor [B,T]  (decoder input)
      - tgt_out: LongTensor [B,T] (next-token labels)
      - tgt_key_padding_mask: BoolTensor [B,T] (True where PAD)
      - tgt_attn_mask: FloatTensor [T,T] causal -inf/0
      - lengths: List[int] original lengths
      - image_ids: List[str]
    """
    def __init__(self, pad_id: int, eos_id: int):
        self.pad_id = pad_id
        self.eos_id = eos_id

    def __call__(self, batch: List[Dict]):
        # Stack images
        images = torch.stack([b["image"] for b in batch], dim=0)  # [B,3,H,W]

        # Pad caption ids
        ids_list: List[List[int]] = [b["caption_ids"] for b in batch]
        lengths = [len(x) for x in ids_list]
        batch_ids = _pad_sequences(ids_list, pad_id=self.pad_id)  # [B,T]

        # Build decoder inputs/targets by shifting
        tgt_in  = batch_ids[:, :-1]                  # [B,T-1]
        tgt_out = batch_ids[:, 1:]                   # [B,T-1]
        # Key padding mask where True means ignore
        tgt_key_padding_mask = _make_padding_mask(tgt_in, pad_id=self.pad_id)  # [B,T-1]
        # Causal attention mask [T-1,T-1]
        tgt_attn_mask = _generate_square_subsequent_mask(tgt_in.size(1))

        image_ids = [b["image_id"] for b in batch]

        return {
            "images": images,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "tgt_key_padding_mask": tgt_key_padding_mask,
            "tgt_attn_mask": tgt_attn_mask,
            "lengths": lengths,
            "image_ids": image_ids,
        }
