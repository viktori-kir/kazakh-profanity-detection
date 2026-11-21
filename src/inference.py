"""Load the saved model and predict on new comments (no retraining)."""
import json
from pathlib import Path

import torch

from model import CharCNN
from textproc import Vocab


class ProfanityClassifier:
    def __init__(self, artifacts_dir="artifacts"):
        d = Path(artifacts_dir)
        self.cfg = json.loads((d / "config.json").read_text())
        self.vocab = Vocab.load(d / "vocab.json")
        self.model = CharCNN(len(self.vocab), self.cfg["emb_dim"], self.cfg["num_filters"],
                             tuple(self.cfg["kernel_sizes"]), self.cfg["dropout"])
        self.model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.threshold = self.cfg["threshold"]

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=256):
        """Probability that each comment contains profanity (class 1)."""
        ids = torch.tensor([self.vocab.encode(t, self.cfg["max_len"]) for t in texts], dtype=torch.long)
        out = [torch.softmax(self.model(ids[i:i + batch_size]), dim=1)[:, 1] for i in range(0, len(ids), batch_size)]
        return torch.cat(out).tolist() if out else []

    def predict(self, texts):
        """List of (label, probability); label 1 = profanity present."""
        return [(int(p >= self.threshold), p) for p in self.predict_proba(texts)]
