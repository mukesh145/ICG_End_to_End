# icg/training/optimizer.py
from __future__ import annotations
import math
from typing import Tuple
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def _warmup_cosine_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    # Cosine decay to 0.1x min scale to avoid total collapse near end
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    scale = 0.1 + 0.9 * cosine
    return max(1e-8, scale)

def build_optimizer_scheduler(
    model: torch.nn.Module,
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    total_steps: int = 100_000,
    warmup_steps: int = 1_000,
) -> Tuple[torch.optim.Optimizer, LambdaLR]:
    # Separate out weight-decay params per common practice
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "norm" not in n.lower():
            decay.append(p)
        else:
            no_decay.append(p)
    optim = AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    lr_lambda = lambda step: _warmup_cosine_lambda(
        step, warmup_steps=warmup_steps, total_steps=total_steps
    )
    scheduler = LambdaLR(optim, lr_lambda=lr_lambda)
    return optim, scheduler
