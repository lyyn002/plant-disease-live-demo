"""Fine-tune MobileNetV3 on a focused PlantVillage crop subset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from scripts.prepare_data import write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

FOCUS_CROPS = {"Tomato", "Potato", "Apple", "Pepper_bell"}
IMAGE_SIZE = 224


class PlantVillageSubset(Dataset):
    """In-memory subset of PlantVillage images."""

    def __init__(self, rows: list[dict], transform: transforms.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = self.transform(image)
        return tensor, row["label_id"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def filter_rows(split_rows: list[dict], crops: set[str], max_per_class: int) -> list[dict]:
    """Keep only selected crops and cap samples per class for faster training."""
    filtered: list[dict] = []
    per_class: Counter[str] = Counter()
    for row in split_rows:
        crop = row["crop"]
        if crop not in crops:
            continue
        label = row["label"]
        if per_class[label] >= max_per_class:
            continue
        per_class[label] += 1
        filtered.append(row)
    return filtered


def build_dataloaders(
    max_per_class: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, list[str], dict[str, dict[str, str]]]:
    """Load PlantVillage and build train/validation loaders."""
    manifest_path = PROJECT_ROOT / "data" / "plantvillage" / "manifest.json"
    if not manifest_path.exists():
        write_manifest(max_per_class=max_per_class)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    train_rows = manifest["train"]
    test_rows = manifest["test"]

    class_names = sorted({row["label"] for row in train_rows})
    label_to_id = {name: index for index, name in enumerate(class_names)}

    for rows in (train_rows, test_rows):
        for row in rows:
            row["label_id"] = label_to_id[row["label"]]

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        PlantVillageSubset(train_rows, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        PlantVillageSubset(test_rows, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    label_metadata = {
        label: {
            "crop": label.split("___", 1)[0].replace("_", " "),
            "disease": label.split("___", 1)[1].replace("_", " ") if "___" in label else "Unknown",
        }
        for label in class_names
    }
    return train_loader, val_loader, class_names, label_metadata


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """Return accuracy and macro F1 on a loader."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    correct = sum(int(pred == label) for pred, label in zip(all_preds, all_labels, strict=True))
    accuracy = correct / len(all_labels)

    f1_scores: list[float] = []
    num_classes = max(all_labels) + 1
    for class_id in range(num_classes):
        tp = sum(int(pred == class_id and label == class_id) for pred, label in zip(all_preds, all_labels, strict=True))
        fp = sum(int(pred == class_id and label != class_id) for pred, label in zip(all_preds, all_labels, strict=True))
        fn = sum(int(pred != class_id and label == class_id) for pred, label in zip(all_preds, all_labels, strict=True))
        if tp == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            continue
        f1_scores.append(2 * precision * recall / (precision + recall))
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    avg_loss = total_loss / len(all_labels)
    return accuracy, macro_f1 if macro_f1 else accuracy, avg_loss


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    train_loader, val_loader, class_names, label_metadata = build_dataloaders(
        max_per_class=args.max_per_class,
        batch_size=args.batch_size,
    )

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(seen, 1)
        val_acc, val_f1, val_loss = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
        )
        if val_f1 >= best_f1:
            best_f1 = val_f1
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "focus_crops": sorted(FOCUS_CROPS),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                },
                MODELS_DIR / "plant_disease_mobilenetv3.pt",
            )
            with (MODELS_DIR / "class_labels.json").open("w", encoding="utf-8") as handle:
                json.dump(label_metadata, handle, indent=2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "best_val_macro_f1": best_f1, "num_classes": len(class_names)}, handle, indent=2)
    print(f"Saved model to {MODELS_DIR / 'plant_disease_mobilenetv3.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
