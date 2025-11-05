from __future__ import annotations
import json
from collections import Counter
from typing import Iterable, List, Dict, Optional


SPECIAL_TOKENS = {
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


class Vocab:
    """
    Minimal word-level vocabulary.
    - Add words from token streams with a frequency threshold.
    - Provides stoi/itos, encode/decode utilities.
    """

    def __init__(
        self,
        tokens: Optional[Iterable[str]] = None,
        min_freq: int = 2,
        specials: Dict[str, str] = SPECIAL_TOKENS,
        max_size: Optional[int] = None,
    ):
        self.min_freq = min_freq
        self.specials = specials.copy()
        self.max_size = max_size

        # Build initial counts if tokens provided; otherwise start empty.
        self.freqs = Counter(tokens) if tokens is not None else Counter()

        # Will be set in build()
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

        # IDs will be available after build()
        self.pad_id = None
        self.bos_id = None
        self.eos_id = None
        self.unk_id = None

    def add_sentence(self, tokens: Iterable[str]) -> None:
        self.freqs.update(tokens)

    def build(self) -> None:
        # Start with specials
        vocab_items = []
        for key in ["pad", "bos", "eos", "unk"]:
            vocab_items.append(self.specials[key])

        # Sort remaining by frequency (desc), then alphabetically for determinism
        words = [
            w for w, c in self.freqs.items() if c >= self.min_freq and w not in vocab_items
        ]
        words.sort(key=lambda w: (-self.freqs[w], w))

        if self.max_size is not None:
            space = max(0, self.max_size - len(vocab_items))
            words = words[:space]

        self.itos = vocab_items + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

        self.pad_id = self.stoi[self.specials["pad"]]
        self.bos_id = self.stoi[self.specials["bos"]]
        self.eos_id = self.stoi[self.specials["eos"]]
        self.unk_id = self.stoi[self.specials["unk"]]


    def __len__(self) -> int:
        return len(self.itos)

    def word2id(self, w: str) -> int:
        return self.stoi.get(w, self.unk_id)

    def id2word(self, i: int) -> str:
        # print(f"length : {len(self.itos)}")
        # print(i)
        return self.itos[i]


    def encode(
        self,
        tokens: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: Optional[int] = None,
        pad_to_max: bool = False,
    ) -> List[int]:
        """
        Convert token list to IDs; optionally add BOS/EOS and pad/truncate.
        """
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend([self.word2id(t) for t in tokens])
        if add_eos:
            ids.append(self.eos_id)

        if max_len is not None:
            if len(ids) > max_len:
                # Keep BOS + as many tokens as fit, and ensure EOS at the end
                if add_eos:
                    # Leave room for EOS
                    keep = max_len - 1
                    ids = ids[:keep]
                    ids[-1] = self.eos_id  # ensure ends with EOS
                else:
                    ids = ids[:max_len]
            elif pad_to_max:
                ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids
    

    def decode(self, ids: List[int], strip_special: bool = True) -> List[str]:
        """
        Convert IDs back to tokens; optionally remove specials.
        """
        tokens = [self.id2word(i) for i in ids]
        if strip_special:
            specials = set(self.specials.values())
            tokens = [t for t in tokens if t not in specials]
        return tokens

    def to_dict(self) -> Dict:
        return {
            "itos": self.itos,
            "freqs": dict(self.freqs),
            "min_freq": self.min_freq,
            "specials": self.specials,
            "max_size": self.max_size,
        }

    @classmethod
    def from_dict(cls, obj: Dict) -> "Vocab":
        v = cls(tokens=None, min_freq=obj["min_freq"], specials=obj["specials"], max_size=obj["max_size"])
        v.itos = obj["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        v.freqs = Counter(obj.get("freqs", {}))
        v.pad_id = v.stoi[v.specials["pad"]]
        v.bos_id = v.stoi[v.specials["bos"]]
        v.eos_id = v.stoi[v.specials["eos"]]
        v.unk_id = v.stoi[v.specials["unk"]]
        return v

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls.from_dict(obj)