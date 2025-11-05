# icg/data/flickr8k_dataset.py
from __future__ import annotations
import json
import os
import random
from typing import List, Dict, Tuple

from PIL import Image
from torch.utils.data import Dataset

from icg.data.vocab import Vocab
from icg.utils.tokenizer import tokenize
from icg.data.transforms import get_train_transforms, get_val_transforms

# -------- Hardcoded data roots (edit to your layout) --------
DATA_ROOT = "./data/Flickr8k"
IMAGES_DIR = os.path.join(DATA_ROOT, "Images")
ARTIFACTS_DIR = os.path.join(DATA_ROOT, "artifacts")   # created by scripts/prepare_flickr8k.py

VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")
TRAIN_JSONL = os.path.join(ARTIFACTS_DIR, "captions_train.jsonl")
VAL_JSONL   = os.path.join(ARTIFACTS_DIR, "captions_val.jsonl")
TEST_JSONL  = os.path.join(ARTIFACTS_DIR, "captions_test.jsonl")

SPLIT2FILE = {
    "train": TRAIN_JSONL,
    "val": VAL_JSONL,
    "test": TEST_JSONL,
}

class Flickr8kDataset(Dataset):
    """
    Expects artifacts produced by scripts/prepare_flickr8k.py:
      - vocab.json
      - captions_{split}.jsonl  (one line per image: {"image": "xxx.jpg", "captions": [["a","man",...], ...]})
    """
    def __init__(self, split: str = "train", max_len: int = 30, sample_strategy: str = "random"):
        assert split in {"train", "val", "test"}
        self.split = split
        self.max_len = max_len
        self.sample_strategy = sample_strategy  # "random" or "first"

        # Load vocab
        self.vocab: Vocab = Vocab.load(VOCAB_PATH)

        # Load jsonl with tokenized captions
        self.samples = self._load_jsonl(SPLIT2FILE[split])

        # Transforms
        self.tf = get_train_transforms() if split == "train" else get_val_transforms()

    def _load_jsonl(self, path: str) -> List[Dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                # obj: {"image": "xxx.jpg", "captions": [["a","man"], ...]}
                samples.append(obj)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _select_caption(self, captions: List[List[str]]) -> List[str]:
        if self.split == "train" and self.sample_strategy == "random":
            return random.choice(captions)
        # deterministic for val/test
        return captions[0] if captions else []

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img_path = os.path.join(IMAGES_DIR, rec["image"])

        # Load image
        with Image.open(img_path).convert("RGB") as img:
            img = self.tf(img)

        # Choose one caption per image for training/eval
        tokens = self._select_caption(rec["captions"])
        # Encode to ids with BOS/EOS and optional truncation/padding handled later in collate
        ids = self.vocab.encode(tokens, add_bos=True, add_eos=True, max_len=self.max_len, pad_to_max=False)

        return {
            "image": img,                   # Tensor [3,H,W]
            "caption_ids": ids,             # List[int]
            "image_id": rec["image"],       # filename
        }
