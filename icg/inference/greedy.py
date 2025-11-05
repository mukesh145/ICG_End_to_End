# icg/inference/greedy.py
from __future__ import annotations
import torch
from typing import List
from icg.data.vocab import Vocab

@torch.no_grad()
def greedy_decode(
    model,
    images: torch.Tensor,     # [B,3,H,W]
    vocab: Vocab,
    device: torch.device,
    max_len: int = 30,
) -> List[str]:
    """
    Greedy decoding returning detokenized strings.
    """
    model.eval()
    memory = model.encode_images(images.to(device))  # [S,B,E]

    B = images.size(0)
    ys = torch.full((B, 1), vocab.bos_id, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        T = ys.size(1)
        # causal mask [T,T]
        attn_mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
        attn_mask = attn_mask.masked_fill(attn_mask, float("-inf")).float()

        # Decode with cached memory
        logits = _forward_with_memory(model, memory, ys, attn_mask)  # [B,T,V]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if torch.all(next_token.squeeze(1) == vocab.eos_id):
            break

    # Convert to strings (strip specials)
    captions = []
    for seq in ys.tolist():
        toks = []
        started = False
        for tid in seq:
            if tid == vocab.bos_id and not started:
                started = True
                continue
            if started:
                if tid == vocab.eos_id:
                    break
                toks.append(vocab.id2word(tid))
        captions.append(" ".join(toks))
    return captions

def _forward_with_memory(model, memory, tgt_in, tgt_attn_mask):
    tgt_in_TB = tgt_in.transpose(0, 1)
    logits_TBV = model.decoder(
        tgt_in=tgt_in_TB,
        memory=memory,
        tgt_pos_enc=model.tgt_pos_enc,
        src_pos_enc=model.src_pos_enc,
        tgt_attn_mask=tgt_attn_mask,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    )
    return logits_TBV.transpose(0, 1).contiguous()
