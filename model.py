import re

import torch
from torch import nn


LABELS = ("низкая", "высокая")
UNK = "<unk>"


def tokenize(text):
    return re.findall(r"[^\W_]+", text.casefold().replace("ё", "е"))


def encode(text, vocab):
    unknown = vocab[UNK]
    return [vocab.get(word, unknown) for word in tokenize(text)] or [unknown]


class ReviewClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, 64, mode="mean")
        self.linear = nn.Linear(64, 2)

    def forward(self, tokens, offsets):
        vectors = self.embedding(tokens, offsets)
        return self.linear(vectors)


def pack_reviews(reviews):
    """offsets[i] — начало i-го отзыва в общем списке токенов."""
    tokens = []
    offsets = []
    for review in reviews:
        offsets.append(len(tokens))
        tokens.extend(review)
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)
