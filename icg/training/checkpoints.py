# icg/training/checkpoints.py
from __future__ import annotations
import os
import torch
from typing import Any, Dict, Optional

def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    best_metric: float,
    ckpt_dir: str = "./checkpoints",
    is_best: bool = False,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    state: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
    }
    last_path = os.path.join(ckpt_dir, "last.pth")
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(ckpt_dir, "best.pth")
        torch.save(state, best_path)
        return best_path
    return last_path

def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return {
        "epoch": state.get("epoch", 0),
        "global_step": state.get("global_step", 0),
        "best_metric": state.get("best_metric", float("-inf")),
    }
