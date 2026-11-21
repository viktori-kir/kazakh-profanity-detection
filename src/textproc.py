"""Text preprocessing and character vocabulary.

Pipeline for one comment:
    raw text -> normalize() -> list of characters -> ids (Vocab.encode) -> fixed length MAX_LEN
"""
import json
import re
import unicodedata
from collections import Counter

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


def normalize(text: str) -> str:
    """Light normalization that keeps the signals profanity relies on.

    - NFKC: unify look-alike Unicode forms (e.g. full-width letters).
    - lowercase: "ГОВНО" and "говно" become the same.
    - letter runs longer than 2 are cut to 2: "бляяяя" -> "бляя", so stretched words
      look like their usual spelling.
    - digits -> "0": numbers carry no profanity signal, this shrinks the vocabulary.
    - whitespace collapsed.
    Punctuation such as "*" and "@" is KEPT: it marks masked words like "с*ка", "з@еб@л".
    """
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"\d", "0", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clip(text: str, max_len: int) -> str:
    """Long comments: keep the beginning and the end (half each).

    Only ~3% of training comments exceed 300 characters; profanity can appear
    anywhere, so keeping both ends loses less than cutting only the tail.
    """
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + text[-(max_len - half):]


class Vocab:
    """Character vocabulary fitted on TRAINING texts only."""

    def __init__(self, chars):
        self.itos = [PAD, UNK] + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_freq=2):
        counts = Counter(ch for t in texts for ch in normalize(t))
        chars = sorted(c for c, n in counts.items() if n >= min_freq)
        return cls(chars)

    def encode(self, text: str, max_len: int):
        """Text -> list of ids, padded with PAD_ID to max_len. Unseen chars -> UNK_ID."""
        t = clip(normalize(text), max_len)
        ids = [self.stoi.get(ch, UNK_ID) for ch in t]
        return ids + [PAD_ID] * (max_len - len(ids))

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos[2:], f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
