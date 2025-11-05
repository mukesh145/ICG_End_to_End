# train.py
from __future__ import annotations
import os
import math
import random
from typing import Dict, Any, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from icg.data.flickr8k_dataset import Flickr8kDataset, ARTIFACTS_DIR as DEFAULT_ARTIFACTS_DIR
from icg.data.collate import CaptionCollate
from icg.data.vocab import Vocab
from icg.models.model import CaptioningModel
from icg.training.optimizer import build_optimizer_scheduler
from icg.training.checkpoints import save_checkpoint
from icg.training.early_stopping import EarlyStopping
from icg.training.loop import train_one_epoch, evaluate_loss_and_bleu

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

import time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# ---------- Config helpers ----------
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _select_device(policy: str) -> torch.device:
    policy = (policy or "auto").lower()
    if policy == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "mps":
        return torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _set_seed(seed: Optional[int], deterministic: bool = False):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN determinism/benchmark
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dicts so we can log to MLflow as params.
    Lists/tuples are converted to comma-joined strings.
    """
    items = []
    for k, v in (d or {}).items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, (list, tuple)):
                v = ",".join([str(x) for x in v])
            items.append((new_key, v))
    return dict(items)



def _load_pretrained_weights_if_any(model: torch.nn.Module, cfg: Dict[str, Any], device: torch.device) -> Optional[str]:
    init_cfg = (cfg.get("mlflow") or {})
    source = str(init_cfg.get("source", "none")).lower()
    strict = bool(init_cfg.get("strict", False))

    # Ensure MLflow tracking is set before trying to load from the registry/runs
    ml_cfg = (cfg.get("mlflow") or {})
    if ml_cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(ml_cfg["tracking_uri"])


    def _try_load_state_dict(sd):
        # Handle {"state_dict": ...} vs direct state dict
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        # Non-strict allows different vocab sizes etc.
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if missing or unexpected:
            print(f"[WARN] load_state_dict(strict={strict}) -> missing={len(missing)}, unexpected={len(unexpected)}")
        model.to(device)


    
    name = ml_cfg.get("model_name")
    stage = init_cfg.get("mlflow_model_stage", "Production")
    if name and stage:
        uri = f"models:/{name}/{stage}"
        try:
            loaded = mlflow.pytorch.load_model(uri)
            _try_load_state_dict(loaded.state_dict())
            return f"mlflow_registry:{uri}"
        except Exception as e:
            print(f"[WARN] Could not load from MLflow registry '{uri}': {e}")


    return None



# ---------- Public entrypoint for Airflow ----------
def train_entrypoint(config: Optional[Dict[str, Any] | str] = None) -> Dict[str, Any]:

    if config is None:
        cfg_path = os.environ.get("CONFIG_PATH", "./icg/config.yaml")
        cfg = _load_yaml(cfg_path)
    elif isinstance(config, str):
        cfg = _load_yaml(config)
    else:
        cfg = config

    # ----- Paths -----
    paths = cfg.get("paths", {})
    artifacts_dir = paths.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR)
    vocab_file = os.path.join(artifacts_dir, paths.get("vocab_file", "vocab.json"))
    ckpt_dir = paths.get("checkpoints_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ----- Device & Backend -----
    device_cfg = cfg.get("device", {})
    device = _select_device(device_cfg.get("policy", "auto"))
    torch.backends.cudnn.benchmark = bool(device_cfg.get("cudnn_benchmark", True))

    # ----- Seed -----
    seed_cfg = cfg.get("seed", {})
    _set_seed(seed_cfg.get("value"), bool(seed_cfg.get("deterministic", False)))

    # ----- Data -----
    data_cfg = cfg.get("data", {})
    max_len = int(data_cfg.get("max_len", 30))
    train_split = data_cfg.get("train", {}).get("split", "train")
    train_sample_strategy = data_cfg.get("train", {}).get("sample_strategy", "random")
    val_split = data_cfg.get("val", {}).get("split", "val")
    val_sample_strategy = data_cfg.get("val", {}).get("sample_strategy", "first")

    # ----- Load vocab -----
    vocab = Vocab.load(vocab_file)

    # ----- Datasets -----
    train_ds = Flickr8kDataset(split=train_split, max_len=max_len, sample_strategy=train_sample_strategy)
    val_ds   = Flickr8kDataset(split=val_split,   max_len=max_len, sample_strategy=val_sample_strategy)

    # ----- Dataloaders -----
    loader_cfg = cfg.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 64))
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", False))
    shuffle_train = bool(loader_cfg.get("shuffle_train", True))
    shuffle_val = bool(loader_cfg.get("shuffle_val", False))

    collate = CaptionCollate(pad_id=vocab.pad_id, eos_id=vocab.eos_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=shuffle_val,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate
    )

    # ----- Model -----
    model_cfg = cfg.get("model", {})

    model = CaptioningModel(
        vocab_size=len(vocab),
        d_model=int(model_cfg.get("d_model", 512)),
        nhead=int(model_cfg.get("nhead", 8)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 2048)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        tie_weights=bool(model_cfg.get("tie_weights", True)),
        encoder_pretrained=bool(model_cfg.get("encoder_pretrained", True)),
    ).to(device)

    print(_load_pretrained_weights_if_any(model, cfg, device=device))

    # ----- Optim & Sched -----
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 1))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 3e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))

    sched_cfg = cfg.get("sched", {})
    warmup_steps = int(sched_cfg.get("warmup_steps", 1000))

    total_steps = max(1, math.ceil(len(train_loader) * epochs / max(1, grad_accum_steps)))
    optimizer, scheduler = build_optimizer_scheduler(
        model, lr=lr, weight_decay=weight_decay, total_steps=total_steps, warmup_steps=warmup_steps
    )

    # ----- Early stopping -----
    es_cfg = cfg.get("early_stopping", {})
    stopper = EarlyStopping(
        patience=int(es_cfg.get("patience", 3)),
        min_delta=0.0,
        mode=str(es_cfg.get("mode", "min")),
    )

    # ----- Train Loop -----
    best_val_loss = float("inf")
    best_epoch = None
    last_metrics = {}

    
    ml_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(ml_cfg.get("tracking_uri", "http://mlflow:5050"))
    mlflow.set_experiment(ml_cfg.get("experiment", "ICG-Project-4.0"))



    with mlflow.start_run(run_name=f"run_{time.time()}"):
        flat_cfg = _flatten_dict(cfg)
        mlflow.log_params(flat_cfg)

        derived_params = {
            "derived.device": str(device),
            "derived.vocab_size": len(vocab),
            "derived.train_size": len(train_ds),
            "derived.val_size": len(val_ds),
            "derived.batch_size": batch_size,
            "derived.num_workers": num_workers,
            "derived.total_steps": total_steps,
            "derived.grad_accum_steps": grad_accum_steps,
        }
        mlflow.log_params(derived_params)


        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} on {device} ===")
            train_loss, _ = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, pad_id=vocab.pad_id
            )

            metrics = evaluate_loss_and_bleu(model, val_loader, device, vocab)

            val_loss = float(metrics["val_loss"])
            val_bleu = float(metrics["bleu"])
            last_metrics = {"train_loss": float(train_loss), "val_loss": val_loss, "bleu": val_bleu}

            print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | BLEU={val_bleu:.2f}")

            # Checkpointing
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch

            
            mlflow.log_metrics(last_metrics, step = epoch)

            if stopper.step(val_loss):
                print("Early stopping triggered.")
                break
          

        mlflow.pytorch.log_model(
            pytorch_model = model,
            artifact_path = "model",
            registered_model_name = ml_cfg.get("model_name")
        )

        run_id = mlflow.active_run().info.run_id
        print('Run ID:',run_id)

        summary = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "last_metrics": last_metrics,
            "checkpoints_dir": ckpt_dir,
            "artifacts_dir": artifacts_dir,
            "device": str(device),
            "run_id":run_id,
            "model_name": ml_cfg.get("model_name")
        }
        print(f"\nTraining finished. Summary: {summary}")
        return summary
    
    mlflow.end_run()



# Optional: keep script executability for local debugging without args.
if __name__ == "__main__":
    train_entrypoint()
