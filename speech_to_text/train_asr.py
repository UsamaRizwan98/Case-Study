import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from utils import get_speechcommands_loaders
from models import SimpleCNN 

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return total_loss / total, correct / total

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labels = ["yes", "no", "up", "down"]
    trainloader, testloader, _ = get_speechcommands_loaders(
        args.data_dir, args.batch_size, labels, n_mels=64
    )

    model = SimpleCNN(num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

        # Save checkpoint for current epoch
        ckpt_path = os.path.join(args.out_dir, f"asr_epoch_{epoch+1}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "labels": labels,
            "model_type": "cnn"
        }, ckpt_path)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(args.out_dir, "asr_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "labels": labels,
                "model_type": "cnn"
            }, best_path)
            print(f"[INFO] Best model updated with acc={best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    main(args)
