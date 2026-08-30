"""Evaluate the fine-tuned model on the held-out PlantVillage test split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_SIZE = 224


def main() -> None:
    manifest_path = PROJECT_ROOT / "data" / "plantvillage" / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        test_rows = json.load(handle)["test"]

    checkpoint = torch.load(MODELS_DIR / "plant_disease_mobilenetv3.pt", map_location="cpu", weights_only=False)
    class_names: list[str] = checkpoint["class_names"]
    label_to_id = {name: index for index, name in enumerate(class_names)}

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for row in test_rows:
            image = Image.open(row["image_path"]).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            pred = model(tensor).argmax(dim=1).item()
            y_true.append(label_to_id[row["label"]])
            y_pred.append(pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()
    summary = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            class_name: {
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1": report[class_name]["f1-score"],
                "support": int(report[class_name]["support"]),
            }
            for class_name in class_names
        },
        "confusion_matrix": matrix,
        "class_names": class_names,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
