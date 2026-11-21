"""Evaluate the saved model ONCE on the held-out test set.

Usage:
    python scripts/evaluate.py --data data/processed --artifacts artifacts --out results
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from inference import ProfanityClassifier  # noqa: E402

# KazSAnDRA id prefix -> source domain (used only for a robustness breakdown)
DOMAIN = {"pla": "appstore", "kas": "market", "fli": "market", "gis": "mapping", "map": "mapping", "kit": "bookstore"}


def scores(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1], average=None, zero_division=0)
    return {"n": int(len(y)), "precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]),
            "accuracy": float(accuracy_score(y, pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(Path(args.data) / "test.csv")
    clf = ProfanityClassifier(args.artifacts)
    test["prob"] = clf.predict_proba(test["text"].tolist())
    test["pred"] = (test["prob"] >= clf.threshold).astype(int)
    test["domain"] = test["kazsandra_id"].str[:3].map(DOMAIN)

    y, pred = test["label"].values, test["pred"].values
    cm = confusion_matrix(y, pred, labels=[0, 1])
    metrics = {"threshold": clf.threshold, "profanity_class": scores(y, pred),
               "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
               "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())}}
    app = test[test["domain"] == "appstore"]
    metrics["appstore_only"] = scores(app["label"].values, app["pred"].values)
    metrics["test_domains"] = test["domain"].value_counts().to_dict()
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["0 clean", "1 profanity"]); ax.set_yticks([0, 1], ["0 clean", "1 profanity"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Test confusion matrix (n={len(test)}, thr={clf.threshold})")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150)

    test["error"] = np.where(test["pred"] == test["label"], "",
                             np.where(test["pred"] == 1, "false_positive", "false_negative"))
    test[["kazsandra_id", "domain", "text", "label", "pred", "prob", "error"]].to_csv(
        out / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
