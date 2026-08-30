"""Model loading and inference for plant leaf disease classification."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from backend.app.config import IMAGE_SIZE


@dataclass
class LabelInfo:
    """Metadata for a single disease class."""

    label: str
    crop: str
    disease: str


def _parse_label(label: str) -> tuple[str, str]:
    """Split PlantVillage label into crop and disease parts."""
    if "___" in label:
        crop, disease = label.split("___", 1)
        return crop.replace("_", " "), disease.replace("_", " ")
    return label, "Unknown"


class PlantDiseaseClassifier:
    """Wraps a fine-tuned MobileNetV3 for inference."""

    def __init__(self, model_path: Path, labels_path: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            pass
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names: list[str] = checkpoint["class_names"]
        num_classes = len(self.class_names)

        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        with labels_path.open(encoding="utf-8") as handle:
            raw_labels = json.load(handle)
        self.label_info = {
            name: LabelInfo(label=name, crop=raw_labels[name]["crop"], disease=raw_labels[name]["disease"])
            for name in self.class_names
        }

        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_bytes: bytes, top_k: int = 3) -> tuple[list[dict[str, object]], float]:
        """Run inference and return ranked predictions plus latency in ms."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        values, indices = torch.topk(probabilities, k=min(top_k, len(self.class_names)))
        predictions: list[dict[str, object]] = []
        for score, index in zip(values.tolist(), indices.tolist(), strict=True):
            class_name = self.class_names[index]
            info = self.label_info[class_name]
            predictions.append(
                {
                    "label": info.label,
                    "crop": info.crop,
                    "disease": info.disease,
                    "confidence": float(score),
                }
            )
        return predictions, elapsed_ms
