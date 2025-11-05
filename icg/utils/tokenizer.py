import regex as re
from typing import List

# Basic normalization rules are intentionally simple and stable.
# They work well for Flickr8k-style English captions.
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT = r"\.\,\!\?\:\;\-\(\)\"\'"
_PUNCT_SEP_RE = re.compile(f"([{_PUNCT}])")
_NON_PRINTING_RE = re.compile(r"[^\p{L}\p{N}\p{Z}\.\,\!\?\:\;\-\(\)\"\'/]")



def normalize_text(text: str) -> str:
    """
    Lowercase, remove non-printing chars, standardize spaces, and
    put spaces around basic punctuation so that splitting is consistent.
    """
    text = text.strip().lower()
    text = _NON_PRINTING_RE.sub(" ", text)
    # Space out punctuation so it's tokenized as separate tokens.
    text = _PUNCT_SEP_RE.sub(r" \1 ", text)
    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Normalize and split on spaces. Keeps punctuation as tokens (.,!?:;-"()' etc.).
    """
    text = normalize_text(text)
    if not text:
        return []
    return text.split()



def detokenize(tokens: List[str]) -> str:
    """
    Simple detokenizer that tries to glue back punctuation nicely.
    """
    if not tokens:
        return ""
    out = []
    for i, tok in enumerate(tokens):
        if tok in {".", ",", "!", "?", ":", ";", ")", "\""}:
            # attach to previous
            if out:
                out[-1] = out[-1] + tok
            else:
                out.append(tok)
        elif tok in {"(", "\""}:
            # attach to next (as prefix)
            out.append(tok)
        elif tok == "-":
            # hyphen: attach to previous and next when possible; fallback to spaced
            if out:
                out[-1] = out[-1] + "-"
            else:
                out.append("-")
        else:
            # regular token: add with space
            out.append(tok if not out else " " + tok)
    return "".join(out).replace("( ", "(").replace(" )", ")").replace("\" ", "\"").replace(" \"", "\"")




