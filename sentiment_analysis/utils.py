import os
import re
import math
import json
import random
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
# Device helper
# -------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -------------------------
# Simple tokenizer & vocab
# -------------------------
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return TOKEN_RE.findall(text)

class Vocab:
    def __init__(self, max_size: int = 20000, min_freq: int = 2, specials: List[str] = None):
        self.max_size = max_size
        self.min_freq = min_freq
        self.freqs = Counter()
        self.itos = []
        self.stoi = {}
        self.specials = specials or ["<pad>", "<unk>"]
        for tok in self.specials:
            self.add_token(tok, count=0)  # reserve positions

    def add_token(self, token, count=1):
        # used internally to reserve special tokens
        if token not in self.freqs:
            self.freqs[token] += count

    def build_from_texts(self, texts: List[str]):
        for t in texts:
            toks = simple_tokenize(t)
            self.freqs.update(toks)

        # filter by min_freq
        items = [(tok, c) for tok, c in self.freqs.items() if c >= self.min_freq and tok not in self.specials]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[: self.max_size - len(self.specials)]
        self.itos = list(self.specials) + [t for t, _ in items]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str], max_len: int = None) -> List[int]:
        ids = [self.stoi.get(t, self.stoi.get("<unk>")) for t in tokens]
        if max_len is not None:
            if len(ids) >= max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.stoi["<pad>"]] * (max_len - len(ids))
        return ids

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"itos": self.itos, "max_size": self.max_size, "min_freq": self.min_freq}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        v = cls(max_size=data.get("max_size", 20000), min_freq=data.get("min_freq", 2))
        v.itos = data["itos"]
        v.stoi = {tok: i for i, tok in enumerate(v.itos)}
        return v

# -------------------------
# Dataset wrapper
# -------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocab, max_len: int = 256):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        toks = simple_tokenize(t)
        ids = self.vocab.encode(toks, max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_batch(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys

# -------------------------
# Basic training utilities
# -------------------------
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

def calc_accuracy(preds: torch.Tensor, targets: torch.Tensor):
    # preds: logits (batch, classes)
    pred_labels = preds.argmax(dim=-1)
    correct = (pred_labels == targets).sum().item()
    return correct / targets.size(0)

def plot_metrics(history: dict, out_path: str):
    # history: {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()