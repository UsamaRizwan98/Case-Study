#!/usr/bin/env python3
import argparse
import torch
import torchaudio
import torchaudio.transforms as T
from models import SimpleCNN  # your actual model

def main(args):
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    labels = ckpt["labels"]
    num_classes = len(labels)

    # Create model and load weights
    model = SimpleCNN(num_classes)
    model.load_state_dict(ckpt["model_state"])

    model.eval()

    # Load and preprocess audio
    waveform, sr = torchaudio.load(args.audio)
    waveform = T.Resample(sr, 16000)(waveform)  # resample to 16k
    mel_spec = T.MelSpectrogram(sample_rate=16000, n_mels=64)(waveform)
    mel_spec = mel_spec.unsqueeze(0)  # add batch dim

    # Run inference
    with torch.no_grad():
        outputs = model(mel_spec)
        pred_idx = outputs.argmax(1).item()
        probs = torch.softmax(outputs, 1)[0].tolist()
        print(f"Predicted: {labels[pred_idx]} (Probs: {probs})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    args = parser.parse_args()
    main(args)