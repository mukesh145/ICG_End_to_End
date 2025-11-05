# icg/training/loop.py
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from icg.utils.metrics import corpus_bleu
from icg.data.vocab import Vocab

# -------- Hardcoded trainer defaults --------
LABEL_SMOOTHING = 0.1
USE_AMP = True
GRAD_CLIP_NORM = 1.0
MAX_GEN_LEN = 30  # for quick val BLEU

def _loss_fn(vocab_size: int, pad_id: int) -> nn.Module:
    return nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING > 0 else 0.0,
        reduction="mean",
    )

@torch.no_grad()
def _greedy_generate(
    model: nn.Module,
    images: torch.Tensor,      # [B,3,H,W]
    vocab: Vocab,
    device: torch.device,
    max_len: int = MAX_GEN_LEN,
) -> List[List[int]]:
    """
    Simple greedy decoding using cached encoder memory.
    Returns list of token id sequences (with BOS..EOS).
    """
    model.eval()
    memory = model.encode_images(images.to(device))  # [S,B,E]
    B = images.size(0)

    # Start with BOS
    ys = torch.full((B, 1), fill_value=vocab.bos_id, dtype=torch.long, device=device)  # [B,1]

    for _ in range(max_len - 1):
        # Prepare masks
        tgt_in = ys  # [B,T]
        T = tgt_in.size(1)
        tgt_attn_mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
        tgt_attn_mask = tgt_attn_mask.masked_fill(tgt_attn_mask, float("-inf")).float()

        logits = model(
            images=None,  # bypass encode (we already did)
            tgt_in=tgt_in,
            tgt_attn_mask=tgt_attn_mask,
            tgt_key_padding_mask=None,
        ) if hasattr(model, "forward_with_memory") else _forward_with_memory(model, memory, tgt_in, tgt_attn_mask)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B,1]
        ys = torch.cat([ys, next_token], dim=1)  # [B,T+1]

        # Stop early if all EOS
        if torch.all(next_token.squeeze(1) == vocab.eos_id):
            break

    return [seq.tolist() for seq in ys]

def _forward_with_memory(model, memory, tgt_in, tgt_attn_mask):
    # Helper that calls decoder directly using model internals.
    # This avoids re-encoding images repeatedly during greedy gen.
    tgt_key_padding_mask = None
    tgt_in_TB = tgt_in.transpose(0, 1)  # [T,B]
    logits_TBV = model.decoder(
        tgt_in=tgt_in_TB,
        memory=memory,
        tgt_pos_enc=model.tgt_pos_enc,
        src_pos_enc=model.src_pos_enc,
        tgt_attn_mask=tgt_attn_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=None,
    )
    return logits_TBV.transpose(0, 1).contiguous()  # [B,T,V]

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    pad_id: int,
) -> Tuple[float, float]:
    model.train()
    scaler = GradScaler(enabled=USE_AMP)
    loss_fn = _loss_fn(vocab_size=model.decoder.vocab_size, pad_id=pad_id)

    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)
        tgt_attn_mask = batch["tgt_attn_mask"].to(device, non_blocking=True)
        tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP, dtype=torch.float16):
            logits = model(
                images=images,
                tgt_in=tgt_in,
                tgt_attn_mask=tgt_attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )  # [B,T,V]
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))

        scaler.scale(loss).backward()
        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        tokens = (tgt_out != pad_id).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        pbar.set_postfix({"loss": f"{(total_loss / max(1,total_tokens)):.4f}"})

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, total_tokens

@torch.no_grad()
def evaluate_loss_and_bleu(
    model: nn.Module,
    dataloader,
    device: torch.device,
    vocab: Vocab,
) -> Dict[str, float]:
    model.eval()
    loss_fn = _loss_fn(vocab_size=model.decoder.vocab_size, pad_id=vocab.pad_id)

    total_loss = 0.0
    total_tokens = 0

    all_refs: List[List[List[str]]] = []
    all_hyps: List[List[str]] = []

    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)
        tgt_attn_mask = batch["tgt_attn_mask"].to(device, non_blocking=True)
        tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device, non_blocking=True)

        with autocast(enabled=USE_AMP, dtype=torch.float16):
            logits = model(
                images=images,
                tgt_in=tgt_in,
                tgt_attn_mask=tgt_attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))

        tokens = (tgt_out != vocab.pad_id).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens

        # quick greedy BLEU
        gen_ids = _greedy_generate(model, images, vocab, device, max_len=MAX_GEN_LEN)
        # Remove BOS, truncate after EOS
        hyps = []
        for seq in gen_ids:
            cleaned = []
            started = False
            for tid in seq:
                if tid == vocab.bos_id and not started:
                    started = True
                    continue
                if started:
                    if tid == vocab.eos_id:
                        break
                    cleaned.append(vocab.id2word(tid))
            hyps.append(cleaned)

        refs = [[vocab.decode(seq.tolist(), strip_special=True)] for seq in tgt_out]  # poor man's ref
        all_refs.extend(refs)
        all_hyps.extend(hyps)

    avg_loss = total_loss / max(1, total_tokens)
    bleu = corpus_bleu(all_refs, all_hyps, max_n=4)
    return {"val_loss": avg_loss, "bleu": bleu}
