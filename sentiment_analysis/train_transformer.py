#!/usr/bin/env python3
"""
Train a simple Transformer encoder classifier on IMDB dataset.
Safe for Apple Silicon MPS (no nested tensor mask bug).
"""

import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import get_device, Vocab, TextDataset, collate_batch, save_checkpoint, plot_metrics
from models import TransformerClassifier
import glob

# -------------------------
# Local IMDB loader
# -------------------------
def load_imdb_fallback(data_dir="aclImdb", split="train"):
    texts, labels = [], []
    for label_name, label_val in [("neg", 0), ("pos", 1)]:
        path_pattern = os.path.join(data_dir, split, label_name, "*.txt")
        for file_path in glob.glob(path_pattern):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label_val)
    return texts, labels

def load_imdb_locally(split="train"):
    try:
        from torchtext.datasets import IMDB
        ds = list(IMDB(root=".data", split=split))
        texts = [t for label, t in ds]
        labels = [0 if label.lower().startswith("neg") else 1 for label, t in ds]
        return texts, labels
    except Exception:
        print("[WARN] torchtext IMDB not available, falling back to local dataset.")
        if not os.path.exists("aclImdb"):
            raise RuntimeError(
                "Local IMDB dataset not found. Please download from:\n"
                "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz\n"
                "and extract into ./aclImdb"
            )
        return load_imdb_fallback(data_dir="aclImdb", split=split)

def split_train_val(texts, labels, val_frac=0.1, seed=42):
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - val_frac))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    return (
        [texts[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [texts[i] for i in val_idx],
        [labels[i] for i in val_idx],
    )

# -------------------------
# Training / Eval loops
# -------------------------
def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)  # No src_key_padding_mask used in model
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(dim=-1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, total_acc / n

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_acc += (logits.argmax(dim=-1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, total_acc / n

# -------------------------
# Main
# -------------------------
def main(args):
    device = get_device()
    print("Using device:", device)

    texts, labels = load_imdb_locally(split="train")
    print("Loaded IMDB train size:", len(texts))

    if args.limit and args.limit > 0:
        texts, labels = texts[:args.limit], labels[:args.limit]

    train_texts, train_labels, val_texts, val_labels = split_train_val(
        texts, labels, val_frac=args.val_frac
    )

    print("Building vocab...")
    vocab = Vocab(max_size=args.vocab_size, min_freq=args.min_freq)
    vocab.build_from_texts(train_texts)
    print("Vocab size:", len(vocab))
    os.makedirs(args.out_dir, exist_ok=True)
    vocab.save(os.path.join(args.out_dir, "vocab.json"))

    train_ds = TextDataset(train_texts, train_labels, vocab, max_len=args.max_len)
    val_ds = TextDataset(val_texts, val_labels, vocab, max_len=args.max_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )

    model = TransformerClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_len=args.max_len,
        num_classes=2,
        dropout=args.dropout,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} -- "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab": vocab.itos,
            "history": history,
        }
        torch.save(ckpt, os.path.join(args.out_dir, f"transformer_epoch{epoch}.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(args.out_dir, "transformer_best.pt"))

    plot_metrics(history, os.path.join(args.out_dir, "transformer_training_metrics.png"))
    print("Training complete. Best val acc:", best_val_acc)

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="checkpoints_transformer")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    main(args)
