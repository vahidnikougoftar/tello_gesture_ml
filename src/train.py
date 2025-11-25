"""Train a CNN to classify drone gestures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import GestureCNN


def get_dataloaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader, dict]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose(
        [
            transforms.Resize((72, 72)),
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_dataset.class_to_idx


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc="val", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / len(loader), correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = get_dataloaders(args.data_dir, args.batch_size)
    model = GestureCNN(num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = args.output_dir / "gesture_cnn.pt"
    log_path = args.output_dir / "class_to_idx.json"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": epoch,
            }
            torch.save(checkpoint, best_path)
            with log_path.open("w", encoding="utf-8") as fp:
                json.dump(class_to_idx, fp, indent=2)
            print(f"[checkpoint] saved to {best_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
