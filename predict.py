"""Определяет оценку по тексту отзыва о клинике."""

import argparse
import json
from pathlib import Path

import torch

from model import LABELS, ReviewClassifier, encode, pack_reviews, tokenize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="", help="Текст отзыва в кавычках")
    args = parser.parse_args()
    if not tokenize(args.text):
        parser.exit(2, "Введите отзыв (хотя бы одно слово или число).\n")

    artifacts = Path(__file__).resolve().parent / "artifacts"
    vocab = json.loads((artifacts / "vocab.json").read_text(encoding="utf-8"))
    model = ReviewClassifier(len(vocab))
    weights = torch.load(artifacts / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    with torch.inference_mode():
        tokens, offsets = pack_reviews([encode(args.text, vocab)])
        prediction = model(tokens, offsets).argmax(dim=1).item()
    print(f"Прогноз оценки: {LABELS[prediction]}")


if __name__ == "__main__":
    main()
