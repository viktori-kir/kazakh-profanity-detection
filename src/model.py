"""Character-level CNN text classifier (Kim-style CNN over characters).

ids (B, L) -> Embedding -> (B, L, E) -> transpose -> (B, E, L)
           -> Conv1d with kernel sizes 3, 4, 5 (in parallel) + ReLU -> (B, F, L')
           -> global max pool over positions -> (B, F) for each kernel size
           -> concat (B, 3F) -> dropout -> Linear -> logits (B, 2)

Why max pooling: a comment is profane if a profane fragment appears ANYWHERE.
Each filter learns to fire on a character n-gram (e.g. a swear-word stem or a masked
variant); max pooling keeps its strongest match regardless of position or comment length.
"""
import torch
import torch.nn as nn

from textproc import PAD_ID


class CharCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=48, num_filters=96, kernel_sizes=(3, 4, 5),
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, ids):
        x = self.embedding(ids).transpose(1, 2)            # (B, E, L)
        mask = (ids != PAD_ID).unsqueeze(1)                # (B, 1, L): True on real characters
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))[:, :, : ids.size(1)]   # (B, F, L)
            h = h.masked_fill(~mask, 0.0)                  # padding cannot win the max (ReLU >= 0)
            pooled.append(h.max(dim=2).values)             # (B, F)
        z = self.dropout(torch.cat(pooled, dim=1))         # (B, 3F)
        return self.fc(z)                                  # (B, 2) raw logits
