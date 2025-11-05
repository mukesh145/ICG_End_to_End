# app.py
from __future__ import annotations

import io
import os
import time
import json
import threading
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

from pathlib import Path

import boto3
from botocore.exceptions import ClientError

import torch
import torch.nn.functional as F

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision import transforms

import sys
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from icg.models.model import CaptioningModel  # noqa: E402

# -----------------------------
# Environment / Config
# -----------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "icg-bucket-mlops")
S3_PREFIX = os.getenv("S3_PREFIX", "Production-model")

S3_MODEL_FILE = os.getenv("S3_MODEL_FILE", "model.pth")
S3_VOCAB_FILE = os.getenv("S3_VOCAB_FILE", "vocab.json")
S3_VERSION_FILE = os.getenv("S3_VERSION_FILE", "version.txt")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "300"))  # 5 minutes
DEVICE = os.getenv("DEVICE", "cpu")                   # "cuda" or "cpu"
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
EOS_TOKEN = os.getenv("EOS_TOKEN", "<eos>")           # must exist in itos
BOS_TOKEN = os.getenv("BOS_TOKEN", "<bos>")
MAX_LEN = int(os.getenv("MAX_LEN", "30"))

# -----------------------------
# Notes on model contract:
# model(images: [B,3,H,W]) -> logits [B, T, V]
# We'll greedy-decode argmax over V until EOS_TOKEN.
# -----------------------------

@dataclass
class LoadedAssets:
    model: Optional[torch.nn.Module]
    itos: List[str]           # index -> token (list)
    eos_token: str
    version: Optional[str]    # version string from version.txt; may be None before first load
    device: str

class VersionInfo(BaseModel):
    version: Optional[str]

_state_lock = threading.RLock()
_state: Optional[LoadedAssets] = None

# -----------------------------
# S3 + FS helpers
# -----------------------------
def _s3_key(filename: str) -> str:
    return f"{S3_PREFIX.rstrip('/')}/{filename}"

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _local_path(filename: str) -> str:
    _ensure_dir(ARTIFACT_DIR)
    return os.path.join(ARTIFACT_DIR, filename)

def _download_to_path(s3, bucket: str, key: str, local_path: str) -> None:
    try:
        _ensure_dir(os.path.dirname(local_path))
        s3.download_file(bucket, key, local_path)
    except ClientError as e:
        raise RuntimeError(f"S3 download failed for s3://{bucket}/{key}: {e}")

def _read_s3_text(s3, bucket: str, key: str, encoding: str = "utf-8") -> str:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode(encoding).strip()
    except ClientError as e:
        raise RuntimeError(f"Failed to read s3://{bucket}/{key}: {e}")

def _fetch_remote_version(s3) -> str:
    """Read version string from S3 version.txt (cheap) WITHOUT touching model/vocab."""
    version = _read_s3_text(s3, S3_BUCKET, _s3_key(S3_VERSION_FILE))
    if not version:
        raise RuntimeError("version.txt is empty.")
    return version

def _download_version_file(s3) -> str:
    """Download version.txt to ARTIFACT_DIR and return its string value."""
    local_ver_path = _local_path(S3_VERSION_FILE)
    _download_to_path(s3, S3_BUCKET, _s3_key(S3_VERSION_FILE), local_ver_path)
    with open(local_ver_path, "r", encoding="utf-8") as f:
        v = f.read().strip()
    if not v:
        raise RuntimeError("Downloaded version.txt is empty.")
    return v

def _download_model_and_vocab(s3) -> Dict[str, str]:
    """Download model.pth and vocab.json to ARTIFACT_DIR. Returns local paths."""
    paths = {}
    for fname in (S3_MODEL_FILE, S3_VOCAB_FILE):
        key = _s3_key(fname)
        local = _local_path(fname)
        _download_to_path(s3, S3_BUCKET, key, local)
        paths[fname] = local
    return paths

# -----------------------------
# Vocab / Model loaders
# -----------------------------
def _token_index(itos: List[str], token: str) -> Optional[int]:
    try:
        return itos.index(token)
    except ValueError:
        return None

def _load_vocab_list(vocab_path: str) -> List[str]:
    """
    Expects vocab.json to contain:
      { "itos": ["<pad>", "<bos>", "<eos>", "a", "man", ...] }
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "itos" not in data or not isinstance(data["itos"], list):
        raise RuntimeError("vocab.json must contain an 'itos' list.")
    return data["itos"]

def _compose_device() -> str:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _load_model(model_path: str, device: str) -> torch.nn.Module:
    """
    Robust loader for PyTorch >=2.6 with custom class.
    We trust our own checkpoints, so we allow unpickling.
    """
    # If available, try allow-listing the custom class for safe loading paths.
    try:
        from torch.serialization import add_safe_globals 
        add_safe_globals([CaptioningModel])
        print("Did safe load")
    except Exception:
        pass
    print("came here")
    try:
        # Use trusted load (weights_only=False) for your own checkpoints
        print(model_path)
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("also this modle")
    except TypeError:
        # Older PyTorch without weights_only parameter
        model = torch.load(model_path, map_location=device)
    print("I am here")
    if isinstance(model, torch.nn.Module):
        model.to(device)
        model.eval()
        print("loaded the model")
        return model

    # If a state_dict was saved instead of full module.
    raise RuntimeError(
        "Loaded object is not a torch.nn.Module. "
        "Ensure you saved the full model, or adapt loader to construct the model and load state_dict."
    )

def _preprocess_image(img: Image.Image, image_size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img.convert("RGB"))

def _decode_logits_to_text(
    logits_seq: torch.Tensor,
    itos: List[str],
    eos_token: str
) -> str:
    """
    logits_seq: [B, T, V] or [T, V] (handles B==1)
    Greedy decode; stop at EOS if found.
    """
    if logits_seq.dim() == 3:
        if logits_seq.size(0) != 1:
            raise ValueError("Single-image endpoint expects batch size 1.")
        logits_seq = logits_seq[0]  # [T, V]

    token_ids = torch.argmax(logits_seq, dim=-1).tolist()
    try:
        eos_idx = itos.index(eos_token)
    except ValueError:
        eos_idx = None

    words: List[str] = []
    V = len(itos)
    for tid in token_ids:
        if eos_idx is not None and tid == eos_idx:
            break
        if 0 <= tid < V:
            words.append(itos[tid])

    # Small cleanup for readability
    s = " ".join(words)
    s = (
        s.replace(" ,", ",")
         .replace(" .", ".")
         .replace(" !", "!")
         .replace(" ?", "?")
         .replace(" '", "'")
    )
    return s.strip()

# -----------------------------
# Global state management
# -----------------------------
def _get_state() -> LoadedAssets:
    global _state
    with _state_lock:
        if _state is None:
            _state = LoadedAssets(model=None, itos=[], eos_token=EOS_TOKEN, version=None, device=_compose_device())
        return _state

def _set_state(new_state: LoadedAssets) -> None:
    global _state
    with _state_lock:
        _state = new_state

    print(_state)

# -----------------------------
# Loaders that implement your policy
# -----------------------------
def _initial_load_from_s3() -> None:
    """
    Startup behavior:
      1) Download version.txt, model.pth, vocab.json into ARTIFACT_DIR
      2) Read version and load model + vocab into memory
    """
    s3 = boto3.client("s3")
    # download version.txt and save locally

    version = _download_version_file(s3)
    print(version)
    # download model + vocab
    paths = _download_model_and_vocab(s3)
    print(paths)
    itos = _load_vocab_list(paths[S3_VOCAB_FILE])
    print("Itos")
    device = _compose_device()
    print("device")
    model = _load_model(paths[S3_MODEL_FILE], device)
    print("model")

    _set_state(LoadedAssets(model=model, itos=itos, eos_token=EOS_TOKEN, version=version, device=device))
    print("set_state")
    print(f"[startup] Loaded version {version} from S3 into memory.")

def _reload_if_version_changed() -> None:
    """
    Poller behavior:
      - Read version.txt from S3
      - If different from local in-memory version, download model & vocab & version.txt and hot-swap
    """
    s3 = boto3.client("s3")
    remote_version = _fetch_remote_version(s3)

    st = _get_state()
    if st.version != remote_version:
        print(f"[poller] Detected new version {remote_version} (local was {st.version}). Updating...")
        _download_model_and_vocab(s3)
        version = _download_version_file(s3)

        itos = _load_vocab_list(_local_path(S3_VOCAB_FILE))
        device = st.device  # keep existing device decision
        model = _load_model((S3_MODEL_FILE), device)

        _set_state(LoadedAssets(model=model, itos=itos, eos_token=EOS_TOKEN, version=version, device=device))
        print(f"[poller] Hot-swapped to version {version}.")
    else:
        # Keep a local copy of version.txt refreshed (optional)
        try:
            _download_version_file(s3)
        except Exception as e:
            print(f"[poller] Skipped refreshing local version.txt: {e}")

# -----------------------------
# Background poller
# -----------------------------
def _poller_loop():
    while True:
        try:
            st = _get_state()
            if st.version is None or st.model is None or not st.itos:
                _initial_load_from_s3()
            else:
                _reload_if_version_changed()
        except Exception as e:
            print(f"[poller] Error: {e}")
        time.sleep(POLL_SECONDS)




def _greedy_decode_with_teacher_forcing_interface(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,   # [B,3,H,W], B must be 1 here
    itos: List[str],
    bos_token: str,
    eos_token: str,
    max_len: int,
    device: str,
) -> str:
    assert image_tensor.size(0) == 1, "Only batch size 1 is supported by this endpoint."

    bos_idx = _token_index(itos, bos_token)
    if bos_idx is None:
        raise RuntimeError(f"BOS token '{bos_token}' not found in vocabulary.")
    eos_idx = _token_index(itos, eos_token)
    # eos_idx can be None; we’ll just never early-stop in that case

    # Start with BOS
    # Shape expected by most captioners: [B, T] of int64 (token ids)
    generated = torch.tensor([[bos_idx]], dtype=torch.long, device=device)  # [1,1]

    # Collect word ids as we go (excluding BOS when rendering)
    out_ids: List[int] = []

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            # Forward with current prefix
            # Expect logits shape [B, T, V]; we take the last timestep
            logits = model(image_tensor, generated)  # <- key change vs before
            if not isinstance(logits, torch.Tensor) or logits.dim() != 3:
                raise RuntimeError("Model must return logits as a Tensor of shape [B,T,V].")

            next_logits = logits[:, -1, :]  # [1, V]
            next_id = int(torch.argmax(next_logits, dim=-1).item())
            out_ids.append(next_id)

            # Early stop on EOS
            if eos_idx is not None and next_id == eos_idx:
                break

            # Append next_id to prefix and continue
            next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            generated = torch.cat([generated, next_token], dim=1)  # [1, t+1]

    # Map to tokens, drop everything after EOS (and drop EOS itself)
    words: List[str] = []
    V = len(itos)
    for tid in out_ids:
        if eos_idx is not None and tid == eos_idx:
            break
        if 0 <= tid < V:
            words.append(itos[tid])

    # small cleanups
    s = " ".join(words)
    s = (
        s.replace(" ,", ",")
         .replace(" .", ".")
         .replace(" !", "!")
         .replace(" ?", "?")
         .replace(" '", "'")
    )
    return s.strip()


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Image Caption Inference API", version="1.3")

@app.on_event("startup")
def _startup():
    try:
        print("[startup] Downloading artifacts and loading model...")
        _initial_load_from_s3()
    except Exception as e:
        # Do not crash; /health will report unhealthy until this recovers
        print(f"[startup] Failed to load from S3: {e}")

    # Start the background poller
    t = threading.Thread(target=_poller_loop, daemon=True)
    t.start()
    print("[startup] Poller started.")

@app.get("/health")
def health():
    try:
        st = _get_state()
        ok = st.model is not None and isinstance(st.itos, list) and len(st.itos) > 0 and st.version is not None
        if ok:
            return {"status": "ok", "version": st.version, "device": st.device}
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "model not loaded"})
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

@app.get("/version", response_model=VersionInfo)
def version():
    st = _get_state()
    return VersionInfo(version=st.version)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ct = (file.content_type or "").lower()
    if ct not in ("image/png", "image/jpg", "image/jpeg", "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {ct}")

    st = _get_state()
    if st.model is None or not st.itos:
        raise HTTPException(status_code=503, detail="Model not yet loaded — please retry shortly.")

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    x = _preprocess_image(img, IMAGE_SIZE).unsqueeze(0).to(st.device)  # [1,3,H,W]

    try:
        caption = _greedy_decode_with_teacher_forcing_interface(
            model=st.model,
            image_tensor=x,
            itos=st.itos,
            bos_token=BOS_TOKEN,
            eos_token=st.eos_token,
            max_len=MAX_LEN,
            device=st.device,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {"caption": caption, "version": st.version}


@app.get("/")
def root():
    return {"message": "Image Caption Inference API. Use POST /predict with an image file."}
