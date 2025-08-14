#!/usr/bin/env python3
"""
Train an RNN-based sentiment classifier from scratch on the IMDB dataset.

Usage:
    python train_sentiment.py --model lstm --epochs 6 --batch_size 64
"""
import os
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import get_device, Vocab, TextDataset, collate_batch, save_checkpoint, plot_metrics, simple_tokenize
from models import RNNClassifier
import glob

def load_imdb_fallback(data_dir="aclImdb", split="train"):
    """
    Loads IMDB dataset from a local folder in the Stanford 'aclImdb' format.
    data_dir: path to 'aclImdb' folder
    split: 'train' or 'test'
    Returns: (texts, labels) where labels are 0=neg, 1=pos
    """
    texts, labels = [], []
    for label_name, label_val in [("neg", 0), ("pos", 1)]:
        path_pattern = os.path.join(data_dir, split, label_name, "*.txt")
        for file_path in glob.glob(path_pattern):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label_val)
    return texts, labels

def load_imdb_locally(split="train"):
    """
    Try torchtext.datasets.IMDB first.
    If unavailable, load from local folder.
    """
    try:
        from torchtext.datasets import IMDB
        ds = list(IMDB(root=".data", split=split))
        texts = [t for label, t in ds]
        labels = [0 if label.lower().startswith("neg") else 1 for label, t in ds]
        return texts, labels
    except Exception as e:
        print("[WARN] torchtext IMDB not available, falling back to local dataset.")
        if not os.path.exists("aclImdb"):
            raise RuntimeError(
                "Local IMDB dataset not found. Please download from "
                "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz "
                "and extract into ./aclImdb"
            )
        return load_imdb_fallback(data_dir="aclImdb", split=split)


def split_train_val(texts, labels, val_frac=0.1, seed=42):
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx)*(1-val_frac))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    return train_texts, train_labels, val_texts, val_labels

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_acc += (logits.argmax(dim=-1) == y).sum().item()
        n += batch_size
    return total_loss / n, total_acc / n

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_acc += (logits.argmax(dim=-1) == y).sum().item()
            n += batch_size
    return total_loss / n, total_acc / n

def main(args):
    device = get_device()
    print("Using device:", device)

    # Load IMDB
    texts, labels = load_imdb_locally(split="train")
    print("Loaded IMDB train size:", len(texts))

    # small subset option for quick experiments
    if args.limit and args.limit > 0:
        texts = texts[:args.limit]
        labels = labels[:args.limit]

    train_texts, train_labels, val_texts, val_labels = split_train_val(texts, labels, val_frac=args.val_frac)

    # build vocab
    print("Building vocab...")
    vocab = Vocab(max_size=args.vocab_size, min_freq=args.min_freq)
    vocab.build_from_texts(train_texts)
    print("Vocab size:", len(vocab))
    os.makedirs(args.out_dir, exist_ok=True)
    vocab.save(os.path.join(args.out_dir, "vocab.json"))

    # datasets and loaders
    train_ds = TextDataset(train_texts, train_labels, vocab, max_len=args.max_len)
    val_ds = TextDataset(val_texts, val_labels, vocab, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # model
    model = RNNClassifier(vocab_size=len(vocab), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, rnn_type=args.rnn_type, bidirectional=args.bidirectional, num_classes=2, dropout=args.dropout)
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
        print(f"Epoch {epoch}/{args.epochs} -- train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        # checkpoint
        ckpt_path = os.path.join(args.out_dir, f"model_epoch{epoch}.pt")
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab": vocab.itos,
            "history": history
        }, ckpt_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab": vocab.itos,
                "history": history
            }, os.path.join(args.out_dir, "best_model.pt"))
    # save final history plot
    plot_metrics(history, os.path.join(args.out_dir, "training_metrics.png"))
    print("Training complete. Best val acc:", best_val_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="output directory")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm","gru"])
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="limit dataset for quick test, 0 = no limit")
    args = parser.parse_args()
    main(args)
