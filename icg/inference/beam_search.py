# icg/inference/beam_search.py
from __future__ import annotations
import torch
from typing import List, Tuple
from icg.data.vocab import Vocab

@torch.no_grad()
def beam_search_decode(
    model,
    images: torch.Tensor,        # [B,3,H,W]
    vocab: Vocab,
    device: torch.device,
    max_len: int = 30,
    beam_size: int = 3,
    length_penalty: float = 0.6,  # Google NMT style
) -> List[str]:
    """
    Batched beam search (simple, loops over batch dimension).
    """
    model.eval()
    memory = model.encode_images(images.to(device))  # [S,B,E]
    B = images.size(0)
    results: List[str] = []

    for b in range(B):
        mem_b = memory[:, b:b+1, :]  # [S,1,E]
        caption = _beam_one(model, mem_b, vocab, device, max_len, beam_size, length_penalty)
        results.append(caption)
    return results

def _beam_one(
    model, memory, vocab: Vocab, device, max_len, beam_size, length_penalty
) -> str:
    BOS, EOS = vocab.bos_id, vocab.eos_id

    beams = [(torch.tensor([[BOS]], device=device, dtype=torch.long), 0.0)]  # (seq[T=1], logprob)
    finished: List[Tuple[torch.Tensor, float]] = []

    for _ in range(max_len - 1):
        new_beams = []
        for seq, logp in beams:
            if seq[0, -1].item() == EOS:
                finished.append((seq, logp))
                continue

            T = seq.size(1)
            attn_mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
            attn_mask = attn_mask.masked_fill(attn_mask, float("-inf")).float()
            logits = _forward_with_memory(model, memory, seq, attn_mask)  # [1,T,V]
            logprobs = torch.log_softmax(logits[0, -1, :], dim=-1)        # [V]

            topk_logp, topk_idx = logprobs.topk(beam_size)
            for k in range(beam_size):
                next_id = topk_idx[k].view(1, 1)
                new_seq = torch.cat([seq, next_id], dim=1)                # [1,T+1]
                new_logp = logp + topk_logp[k].item()
                new_beams.append((new_seq, new_logp))

        # prune
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        # if all beams ended
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            finished.extend(beams)
            break

    if not finished:
        finished = beams

    # length-penalized score: lp = ((5+len)/6)^alpha
    def score(seq, logp):
        T = seq.size(1)
        lp = ((5.0 + T) / 6.0) ** length_penalty
        return logp / lp

    best_seq, _ = max(finished, key=lambda x: score(x[0], x[1]))
    # Convert to string
    toks = []
    started = False
    for tid in best_seq[0].tolist():
        if tid == vocab.bos_id and not started:
            started = True
            continue
        if started:
            if tid == vocab.eos_id:
                break
            toks.append(vocab.id2word(tid))
    return " ".join(toks)

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
