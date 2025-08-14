#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_cifar10_loaders
from models import SmallResNet

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Using device:", device)

    trainloader, testloader = get_cifar10_loaders(batch_size=args.batch_size)

    model = SmallResNet(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        # train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        train_acc = correct / total

        # val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{args.epochs}, "
              f"Train Loss: {total_loss/total:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.out_dir, "resnet_best.pt")
            torch.save({
                "model_type": "resnet",
                "model_config": {"num_classes": args.num_classes},
                "model_state": model.state_dict(),
                "val_acc": val_acc
            }, ckpt_path)
            print(f"[INFO] Saved new best model to {ckpt_path} (Val Acc: {val_acc:.4f})")

        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="checkpoints_resnet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()
    main(args)
