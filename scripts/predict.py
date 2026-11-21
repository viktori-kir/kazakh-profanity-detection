"""Predict whether Kazakh comments contain profanity.

Usage:
    python scripts/predict.py "Тауар өте жақсы, тез жеткізді"
    python scripts/predict.py "comment one" "comment two"
    python scripts/predict.py            # interactive: type comments, empty line to quit
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from inference import ProfanityClassifier  # noqa: E402


def show(clf, text):
    label, prob = clf.predict([text])[0]
    name = "PROFANITY" if label else "clean"
    print(f"{label} ({name})  p(profanity)={prob:.3f}  threshold={clf.threshold}  | {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()
    clf = ProfanityClassifier(args.artifacts)
    if args.texts:
        for t in args.texts:
            show(clf, t)
        return
    print("Type a comment and press Enter (empty line to quit).")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        show(clf, line)


if __name__ == "__main__":
    main()
