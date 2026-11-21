"""Train the character CNN.

Usage:
    python scripts/train.py --config configs/train.json --data data/processed --out artifacts

Uses train.csv to fit weights and the character vocabulary, val.csv to pick the best
epoch (early stopping on validation F1 of the profanity class) and the decision threshold.
test.csv is NOT read here.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import CharCNN  # noqa: E402
from textproc import Vocab  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def encode(df, vocab, max_len):
    X = torch.tensor([vocab.encode(t, max_len) for t in df["text"]], dtype=torch.long)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    return X, y


@torch.no_grad()
def predict_proba(model, X, batch_size=256):
    model.eval()
    probs = [torch.softmax(model(X[i:i + batch_size]), dim=1)[:, 1] for i in range(0, len(X), batch_size)]
    return torch.cat(probs).numpy()


def f1_at(probs, y, threshold):
    pred = probs >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.json")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    set_seed(cfg["seed"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(Path(args.data) / "train.csv")
    val_df = pd.read_csv(Path(args.data) / "val.csv")

    vocab = Vocab.build(train_df["text"], min_freq=cfg["min_char_freq"])   # train only
    Xtr, ytr = encode(train_df, vocab, cfg["max_len"])
    Xva, yva = encode(val_df, vocab, cfg["max_len"])
    print(f"vocab size {len(vocab)} | train {tuple(Xtr.shape)} | val {tuple(Xva.shape)}")

    model = CharCNN(len(vocab), cfg["emb_dim"], cfg["num_filters"], tuple(cfg["kernel_sizes"]), cfg["dropout"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(cfg["seed"])

    best_f1, best_epoch, bad_epochs, log = -1.0, 0, 0, []
    t0 = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        total = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        train_loss = total / len(Xtr)

        p_val = predict_proba(model, Xva)
        with torch.no_grad():
            model.eval()
            val_loss = loss_fn(model(Xva), yva).item()
        val_f1 = f1_at(p_val, yva.numpy(), 0.5)
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1@0.5": val_f1})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  val F1@0.5 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"early stopping: no val F1 improvement for {cfg['patience']} epochs")
                break

    # Decision threshold: chosen on validation with the best checkpoint
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    p_val = predict_proba(model, Xva)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [f1_at(p_val, yva.numpy(), t) for t in grid]
    threshold = float(grid[int(np.argmax(f1s))])
    print(f"best epoch {best_epoch} | val F1@0.5 {best_f1:.4f} | chosen threshold {threshold} "
          f"(val F1 {max(f1s):.4f}) | {time.time() - t0:.0f}s")

    vocab.save(out / "vocab.json")
    model_cfg = {k: cfg[k] for k in ["max_len", "emb_dim", "num_filters", "kernel_sizes", "dropout"]}
    model_cfg.update({"threshold": threshold, "best_epoch": best_epoch, "vocab_size": len(vocab),
                      "trainable_parameters": n_params, "seed": cfg["seed"]})
    (out / "config.json").write_text(json.dumps(model_cfg, indent=2))
    pd.DataFrame(log).to_csv(out / "training_log.csv", index=False)


if __name__ == "__main__":
    main()
