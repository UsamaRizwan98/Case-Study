#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F

from utils import get_device, Vocab, simple_tokenize
from models import RNNClassifier, TransformerClassifier

def detect_model_type(state_dict):
    if any(k.startswith("rnn.") for k in state_dict.keys()):
        return "rnn"
    elif any(k.startswith("transformer.") for k in state_dict.keys()):
        return "transformer"
    else:
        raise ValueError("Cannot detect model type from checkpoint keys")

def main(args):
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)

    # rebuild vocab
    vocab = Vocab()
    vocab.itos = ckpt["vocab"]
    vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}

    model_type = detect_model_type(ckpt["model_state"])
    print(f"[INFO] Detected model type: {model_type}")

    # crude guessing of config from weights
    if model_type == "rnn":
        fc_in_features = ckpt["model_state"]["fc.0.weight"].shape[1]
        bidirectional = "rnn.weight_ih_l0_reverse" in ckpt["model_state"]
        hidden_dim = fc_in_features // (2 if bidirectional else 1)
        model = RNNClassifier(
            vocab_size=len(vocab),
            embed_dim=ckpt["model_state"]["embedding.weight"].shape[1],
            hidden_dim=hidden_dim,
            num_layers=1,  # default guess
            rnn_type="lstm",
            bidirectional=bidirectional,
            num_classes=2,
            dropout=0.2
        )
    else:  # transformer
        embed_dim = ckpt["model_state"]["embedding.weight"].shape[1]
        # assume nhead=4, dim_feedforward=256, num_layers=2 (matches your training default)
        model = TransformerClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=256,
            max_len=256,
            num_classes=2,
            dropout=0.1
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # tokenize & encode
    toks = simple_tokenize(args.text)
    ids = vocab.encode(toks, max_len=256)
    inp = torch.tensor([ids], dtype=torch.long).to(device)

    # inference
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())

    label_map = {0: "negative", 1: "positive"}
    print(f"Input: {args.text}")
    print(f"Predicted: {label_map[pred]} (probs: {probs.tolist()})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    main(args)
