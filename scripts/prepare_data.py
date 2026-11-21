"""Step 1: deduplicate, group near-duplicates and split the labeled dataset.

Usage:
    python scripts/prepare_data.py --input data/labeled.csv --out data/processed

Steps
1. Load data/labeled.csv (public, already privacy-cleaned by scripts/privacy_review.py).
2. (label fixes from data/label_corrections.csv are applied after step 5)
3. Drop exact duplicates after normalization.
4. Group near-duplicates (char 3-5-gram TF-IDF cosine >= 0.9 on letters-only text),
   so every group lands in exactly one split.
5. Stratified, group-aware split 80 / 10 / 10 with a fixed seed.
6. Save train/val/test CSVs and a JSON file with split sizes and class counts.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
NEAR_DUP_THRESHOLD = 0.9

def dedup_key(text: str) -> str:
    """Letters-only, lowercased form used to find duplicates and near-duplicates."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\s]|_|\d", " ", t)          # drop punctuation, emoji, digits
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)         # long letter repeats: "керемееет" -> "керемеет"
    return re.sub(r"\s+", " ", t).strip()


def near_duplicate_groups(keys: pd.Series, threshold: float) -> np.ndarray:
    """Union-find over pairs whose char n-gram cosine similarity >= threshold."""
    filled = keys.replace("", "<empty>")
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(filled)
    S = cosine_similarity(X, dense_output=False).tocoo()
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, s in zip(S.row, S.col, S.data):
        if a < b and s >= threshold:
            parent[find(a)] = find(b)
    return np.array([find(i) for i in range(len(keys))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--corrections", default="data/label_corrections.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    n_raw = len(df)
    df["key"] = df["text"].map(dedup_key)

    df = df.drop_duplicates(subset="key", keep="first").reset_index(drop=True)
    n_after_exact = len(df)

    df["group"] = near_duplicate_groups(df["key"], NEAR_DUP_THRESHOLD)
    n_groups_multi = int((df["group"].value_counts() > 1).sum())

    # 10 stratified, group-aware folds: fold 0 -> test, fold 1 -> val, rest -> train
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    df["split"] = "train"
    folds = list(sgkf.split(df, df["label"], groups=df["group"]))
    df.loc[folds[0][1], "split"] = "test"
    df.loc[folds[1][1], "split"] = "val"

    # sanity: no near-duplicate group crosses splits
    assert (df.groupby("group")["split"].nunique() == 1).all()

    # Label fixes are applied AFTER the split, so the split (and the test set) stays the
    # same as with the original labels; only the corrected labels change.
    fixes = pd.read_csv(args.corrections) if Path(args.corrections).exists() else pd.DataFrame()
    for row in fixes.itertuples():
        mask = df["kazsandra_id"] == row.kazsandra_id
        assert mask.sum() == 1 and (df.loc[mask, "label"] == row.old_label).all(), row.kazsandra_id
        df.loc[mask, "label"] = row.new_label

    cols = ["kazsandra_id", "text", "label"]
    stats = {
        "seed": SEED,
        "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
        "rows_labeled": n_raw,
        "label_corrections": int(len(fixes)),
        "rows_after_exact_dedup": n_after_exact,
        "near_duplicate_groups_with_2plus_rows": n_groups_multi,
        "splits": {},
    }
    for name in ["train", "val", "test"]:
        part = df[df["split"] == name]
        part[cols].to_csv(out / f"{name}.csv", index=False)
        stats["splits"][name] = {
            "rows": int(len(part)),
            "label_0": int((part["label"] == 0).sum()),
            "label_1": int((part["label"] == 1).sum()),
        }
    with open(out / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
