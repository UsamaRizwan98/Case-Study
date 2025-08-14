#!/usr/bin/env python3
import os
import torchvision

DATA_DIR = "./data/cifar10"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[INFO] Downloading CIFAR-10 dataset to {DATA_DIR} ...")
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    print("[INFO] CIFAR-10 dataset downloaded successfully!")

if __name__ == "__main__":
    main()