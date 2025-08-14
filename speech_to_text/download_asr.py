#!/usr/bin/env python3
"""
Download a small ASR dataset from HuggingFace and save locally
so train_asr.py can use it without re-downloading.
"""

import os
from datasets import load_dataset
import torchaudio

OUT_DIR = "./data/asr"
DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"  # change to "librispeech_asr" for full set
SPLITS = ["train", "validation"]

def save_wav(audio, sample_rate, path):
    torchaudio.save(path, audio, sample_rate)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in SPLITS:
        split_dir = os.path.join(OUT_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        print(f"[INFO] Downloading {DATASET_NAME} split: {split}")
        dataset = load_dataset(DATASET_NAME, split=split)

        for i, item in enumerate(dataset):
            audio = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            text = item["text"]

            wav_path = os.path.join(split_dir, f"{i}.wav")
            txt_path = os.path.join(split_dir, f"{i}.txt")

            save_wav(torchaudio.tensor(audio).unsqueeze(0), sr, wav_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

        print(f"[INFO] Saved {len(dataset)} samples to {split_dir}")

if __name__ == "__main__":
    main()
