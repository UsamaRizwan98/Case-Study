# CIFAR-10 Image Classification with ResNet & ViT

This section implements two deep learning models — **ResNet** and **Vision Transformer (ViT)** — trained **from scratch** on the **CIFAR-10** dataset. The training and inference pipelines are modular and Colab-ready.

## 📂 Project Structure

.
├── data.py                  # CIFAR-10 download, transforms, dataloaders
├── utils.py                 # Training loop, metrics, checkpoints, logging (summary)
├── models.py                # ResNet & ViT implementations
├── train_resnet.py          # ResNet training pipeline (CLI-friendly)
├── train_vit.py             # ViT training pipeline (CLI-friendly)
├── infer_image.py           # Inference script for single images / small batches
├── cifar10_resnet_vit_consolidated.ipynb  # Colab consolidated notebook
└── README.md                # This file

## Requirement File

pip install -r requirements.txt 

## 📊 Dataset: CIFAR-10
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **60,000 images** (32x32): 50k train / 10k test

### Download & Prepare (automatic in code)
- Uses `torchvision.datasets.CIFAR10` with standard train/test splits.
- Common augmentations: random crop, horizontal flip (customizable in `data.py`).

## 🧠 Models
### ResNet
- Standard residual blocks with skip connections.
- Depth and width are configurable.

### Vision Transformer (ViT)
- Patch embedding + Transformer encoder blocks.
- Configurable patch size, heads, depth, and MLP dim.

## 🚀 Training
### ResNet

python train_resnet.py --epochs 24 --batch_size 128 --lr 0.001 --out_dir checkpoints_resnet


### ViT

python train_vit.py --epochs 24 --batch_size 128 --lr 0.001 --out_dir checkpoints_vit


## 🔍 Inference

for RESNET

python infer_image.py --ckpt checkpoints_resnet/best.pt --image_path ./my_image.jpg

for ViT:

python infer_image.py --ckpt checkpoints_vit/best.pt --image_path ./my_image.jpg




