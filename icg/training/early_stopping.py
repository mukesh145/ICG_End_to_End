# icg/training/early_stopping.py
from __future__ import annotations

class EarlyStopping:
    """
    Simple early stopper on a metric (e.g., val_loss decreasing).
    Set mode='min' for loss, mode='max' for BLEU.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        assert mode in {"min", "max"}
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.count = 0

    def step(self, value: float) -> bool:
        """
        Returns True if should stop.
        """
        if self.best is None:
            self.best = value
            return False

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience
