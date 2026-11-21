"""Step 0 (run once by the dataset author): remove personal information.

Usage:
    python scripts/privacy_review.py --input data/private/kazakh_profanity_share.csv \
        --rules data/private/privacy_rules.json --out data/labeled.csv

Input: the author's labeled file (kazsandra_id, text, label, split). It still contains
private names, so it and the rules file live in data/private/, which is NOT committed.
Output: data/labeled.csv, the public labeled dataset used by every later step.

What it does
1. Latin @handles and _handles_ -> "@user". Cyrillic after "@" is kept: it is masked
   profanity ("қот@қ", "з@еб@**л"), which the model must see.
2. Removes rows listed in rules["exclude"] (they identify or sexualize private people,
   some of them children).
3. In rows listed in rules["redact"], replaces each regex match (names of reviewers,
   staff, doctors, friends; one home address) with "[NAME]". Public figures are kept.
The rules were made by a keyword search (names, ages, schools, phone numbers, surname
endings) followed by manual reading of every hit.
"""
import argparse
import json
import re

import pandas as pd


def scrub_handles(text: str) -> str:
    text = re.sub(r"@(?!user\b)[A-Za-z0-9_.]{2,}", "@user", text)
    text = re.sub(r"(?<!\w)_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_(?!\w)", "@user", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--out", default="data/labeled.csv")
    args = ap.parse_args()

    rules = json.load(open(args.rules, encoding="utf-8"))
    df = pd.read_csv(args.input)
    n_raw = len(df)
    df["text"] = df["text"].astype(str).str.strip().map(scrub_handles)

    df = df[~df["kazsandra_id"].isin(rules["exclude"])].copy()
    for row_id, patterns in rules["redact"].items():
        mask = df["kazsandra_id"] == row_id
        assert mask.sum() == 1, f"redaction id not found: {row_id}"
        for pat in patterns:
            df.loc[mask, "text"] = df.loc[mask, "text"].str.replace(pat, "[NAME]", regex=True)

    df = df.rename(columns={"split": "author_split"})
    df[["kazsandra_id", "text", "label", "author_split"]].to_csv(args.out, index=False)
    print(f"rows in: {n_raw} | removed: {n_raw - len(df)} | rows with names redacted: "
          f"{len(rules['redact'])} | rows out: {len(df)} -> {args.out}")


if __name__ == "__main__":
    main()
