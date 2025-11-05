# icg/utils/metrics.py
from __future__ import annotations
from collections import Counter
from typing import List, Sequence
import math

def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def _modified_precision(references: List[List[str]], hypothesis: List[str], n: int) -> float:
    hyp_counts = _ngram_counts(hypothesis, n)
    if not hyp_counts:
        return 0.0
    max_ref_counts = Counter()
    for ref in references:
        ref_counts = _ngram_counts(ref, n)
        for ng, c in ref_counts.items():
            max_ref_counts[ng] = max(max_ref_counts[ng], c)

    clipped = {ng: min(count, max_ref_counts.get(ng, 0)) for ng, count in hyp_counts.items()}
    return sum(clipped.values()) / max(1, sum(hyp_counts.values()))

def _closest_ref_length(references: List[List[str]], hyp_len: int) -> int:
    return min(references, key=lambda r: (abs(len(r) - hyp_len), len(r))).__len__()

def corpus_bleu(
    list_of_references: List[List[List[str]]],  # per-sample: list of refs (each a token list)
    hypotheses: List[List[str]],
    max_n: int = 4,
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    assert len(list_of_references) == len(hypotheses)
    p_ns = [0.0] * max_n
    tiny = 1e-16

    hyp_len_sum = 0
    ref_len_sum = 0

    # Sum log precisions over the corpus
    for refs, hyp in zip(list_of_references, hypotheses):
        hyp_len = len(hyp)
        hyp_len_sum += hyp_len
        ref_len_sum += _closest_ref_length(refs, hyp_len)

        for n in range(1, max_n + 1):
            p = _modified_precision(refs, hyp, n)
            p_ns[n - 1] += math.log(p + tiny)

    # Average log-precisions
    p_ns = [p / max(1, len(hypotheses)) for p in p_ns]

    # Brevity penalty
    if hyp_len_sum > ref_len_sum:
        bp = 1.0
    else:
        bp = math.exp(1 - (ref_len_sum + tiny) / (hyp_len_sum + tiny))

    bleu = bp * math.exp(sum(w * p for w, p in zip(weights, p_ns)))
    return float(bleu * 100.0)  # return BLEU as percentage (0..100)
