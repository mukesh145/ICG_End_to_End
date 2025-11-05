# scripts/prepare_flickr8k.py
"""
Prepare Flickr8k-style artifacts using a YAML config.

Expected config keys:
- paths.data_root, paths.images_subdir, paths.captions_file, paths.artifacts_dir, paths.out.*
- data_prep.{train_size,val_size,test_size,min_freq,max_vocab_size,delimiter,strict_malformed,image_exts,seed}

Usage in Airflow:
    from scripts.prepare_flickr8k import prepare_entrypoint
    @task
    def prep_task():
        return prepare_entrypoint()  # uses CONFIG_PATH or ./config.yaml
"""

from __future__ import annotations
import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import yaml

from icg.utils.tokenizer import tokenize
from icg.data.vocab import Vocab


# ---------------- Config helpers ----------------
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _resolve_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    paths = cfg.get("paths", {})
    data_root = paths.get("data_root", "./data/Flickr8k")

    images_dir = os.path.join(data_root, paths.get("images_subdir", "Images"))
    captions_path = os.path.join(data_root, paths.get("captions_file", "captions.txt"))

    artifacts_dir = paths.get("artifacts_dir", os.path.join(data_root, "artifacts"))
    os.makedirs(artifacts_dir, exist_ok=True)

    out = paths.get("out", {})
    vocab_path = os.path.join(artifacts_dir, out.get("vocab", "vocab.json"))
    train_jsonl = os.path.join(artifacts_dir, out.get("train_jsonl", "captions_train.jsonl"))
    val_jsonl   = os.path.join(artifacts_dir, out.get("val_jsonl", "captions_val.jsonl"))
    test_jsonl  = os.path.join(artifacts_dir, out.get("test_jsonl", "captions_test.jsonl"))

    return {
        "images_dir": images_dir,
        "captions_path": captions_path,
        "artifacts_dir": artifacts_dir,
        "vocab_path": vocab_path,
        "train_jsonl": train_jsonl,
        "val_jsonl": val_jsonl,
        "test_jsonl": test_jsonl,
    }

# ---------------- Core helpers (parametrized) ----------------
def _scan_images(images_dir: str, image_exts: List[str]) -> set:
    exts = {e.lower() for e in image_exts}
    imgs = set()
    for fn in os.listdir(images_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext in exts:
            imgs.add(fn)
    return imgs

def _read_captions(path: str, delimiter: str, strict_malformed: bool) -> Dict[str, List[str]]:
    """
    Read lines of form:  image.jpg<DELIM>caption...
    Returns: { "image.jpg": [cap1, cap2, ...] }
    """
    img2caps = defaultdict(list)
    line_num = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            s = line.strip()
            if not s:
                continue
            parts = s.split(delimiter, 1)  # split only once, exact delimiter
            if len(parts) != 2:
                if strict_malformed:
                    raise ValueError(f"Malformed line {line_num}: {s}")
                # skip malformed if not strict
                continue
            img, cap = parts[0].strip(), parts[1].strip()
            if not img or not cap:
                if strict_malformed:
                    raise ValueError(f"Empty image or caption at line {line_num}")
                continue
            img2caps[img].append(cap)
    return img2caps

def _tokenize_records(img2raw: Dict[str, List[str]], keep_set: set) -> List[Dict[str, Any]]:
    """
    Keep only images that exist on disk and have ≥1 valid caption.
    Deduplicate exact duplicate captions per image.
    """
    records = []
    for img, caps in img2raw.items():
        if img not in keep_set:
            continue
        seen = set()
        toks_list = []
        for c in caps:
            if c in seen:
                continue
            seen.add(c)
            toks = tokenize(c)
            if toks:
                toks_list.append(toks)
        if toks_list:
            records.append({"image": img, "captions": toks_list})
    return records

def _write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _split_records(
    records: List[Dict[str, Any]],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)
    recs = records[:]  # copy
    random.shuffle(recs)

    n = len(recs)
    t_train = min(train_size, n)
    t_val   = min(val_size, max(0, n - t_train))
    t_test  = min(test_size, max(0, n - t_train - t_val))

    train = recs[:t_train]
    val   = recs[t_train:t_train + t_val]
    test  = recs[t_train + t_val:t_train + t_val + t_test]
    return train, val, test

# ---------------- Airflow-friendly entrypoint ----------------
def prepare_entrypoint(config: Optional[Dict[str, Any] | str] = None) -> Dict[str, Any]:
    """
    - If `config` is None: load YAML from os.environ['CONFIG_PATH'] or './config.yaml'
    - If `config` is str: treat as YAML path
    - If `config` is dict: use directly
    Returns a small summary dict.
    """
    # --- Load config ---
    if config is None:
        cfg_path = os.environ.get("CONFIG_PATH", "/Users/user/Desktop/image-caption-generator/icg/config.yaml")
        cfg = _load_yaml(cfg_path)
    elif isinstance(config, str):
        cfg = _load_yaml(config)
    else:
        cfg = config

    paths = _resolve_paths(cfg)
    prep = cfg.get("data_prep", {})

    train_size = int(prep.get("train_size", 6000))
    val_size   = int(prep.get("val_size", 1000))
    test_size  = int(prep.get("test_size", 1000))

    min_freq = int(prep.get("min_freq", 2))
    max_vocab_size = prep.get("max_vocab_size", None)
    delimiter = str(prep.get("delimiter", ","))
    strict_malformed = bool(prep.get("strict_malformed", False))
    image_exts = list(prep.get("image_exts", [".jpg", ".jpeg", ".png"]))
    seed = int(prep.get("seed", 1337))

    # --- Sanity checks ---
    if not os.path.isdir(paths["images_dir"]):
        raise FileNotFoundError(f"Images directory not found: {paths['images_dir']}")
    if not os.path.isfile(paths["captions_path"]):
        raise FileNotFoundError(f"Captions file not found: {paths['captions_path']}")

    # --- Load & parse ---
    available_images = _scan_images(paths["images_dir"], image_exts)  # set of filenames
    img2raw = _read_captions(paths["captions_path"], delimiter, strict_malformed)

    # --- Tokenize & filter ---
    all_records = _tokenize_records(img2raw, keep_set=available_images)
    if len(all_records) == 0:
        raise RuntimeError(
            f"No valid (image, captions) pairs found. Check filenames and delimiter '{delimiter}'."
        )

    # --- Split ---
    train_recs, val_recs, test_recs = _split_records(
        all_records, train_size, val_size, test_size, seed
    )

    # --- Build vocab on TRAIN ONLY ---
    vocab = Vocab(min_freq=min_freq, max_size=max_vocab_size)
    for rec in train_recs:
        for cap_tokens in rec["captions"]:
            vocab.add_sentence(cap_tokens)
    vocab.build()
    vocab.save(paths["vocab_path"])

    # --- Write artifacts ---
    _write_jsonl(train_recs, paths["train_jsonl"])
    _write_jsonl(val_recs,   paths["val_jsonl"])
    _write_jsonl(test_recs,  paths["test_jsonl"])

    kept = {rec["image"] for rec in (train_recs + val_recs + test_recs)}
    summary = {
        "artifacts_dir": paths["artifacts_dir"],
        "vocab_path": paths["vocab_path"],
        "train_jsonl": paths["train_jsonl"],
        "val_jsonl": paths["val_jsonl"],
        "test_jsonl": paths["test_jsonl"],
        "vocab_size": len(vocab),
        "split_counts": {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
        },
        "images_on_disk": len(available_images),
        "images_used": len(kept),
        "delimiter": delimiter,
    }

    print(
        f"Artifacts written to: {paths['artifacts_dir']}\n"
        f"- Vocab size      : {len(vocab)}\n"
        f"- Train/Val/Test  : {len(train_recs)}/{len(val_recs)}/{len(test_recs)}\n"
        f"- Images on disk  : {len(available_images)}\n"
        f"- Images used     : {len(kept)}\n"
        f"- Delimiter       : {delimiter}"
    )
    return summary


# Optional debug run (no args; uses CONFIG_PATH or ./config.yaml)
if __name__ == "__main__":
    prepare_entrypoint()
