"""Benchmark inference latency over repeated runs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from backend.app.model import PlantDiseaseClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def percentile(values: list[float], pct: float) -> float:
    """Return the pct-th percentile from a sorted list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def load_sample_images(num_images: int) -> list[bytes]:
    """Load real PlantVillage test images as JPEG bytes."""
    manifest_path = PROJECT_ROOT / "data" / "plantvillage" / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    images: list[bytes] = []
    for row in manifest["test"]:
        image = Image.open(row["image_path"]).convert("RGB")
        buffer = __import__("io").BytesIO()
        image.save(buffer, format="JPEG")
        images.append(buffer.getvalue())
        if len(images) >= num_images:
            break
    if len(images) < num_images:
        raise RuntimeError(f"Could only load {len(images)} benchmark images.")
    return images


def benchmark(args: argparse.Namespace) -> dict[str, float | int]:
    model_path = PROJECT_ROOT / "models" / "plant_disease_mobilenetv3.pt"
    labels_path = PROJECT_ROOT / "models" / "class_labels.json"
    classifier = PlantDiseaseClassifier(model_path, labels_path)

    sample_images = load_sample_images(args.num_images)
    latencies: list[float] = []

    # Warm-up
    for image_bytes in sample_images[:3]:
        classifier.predict(image_bytes)

    for _ in range(args.runs):
        image_bytes = sample_images[len(latencies) % len(sample_images)]
        _, latency_ms = classifier.predict(image_bytes)
        latencies.append(latency_ms)

    summary = {
        "device": str(classifier.device),
        "runs": args.runs,
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "latency_benchmark.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "all_ms": [round(value, 2) for value in latencies]}, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--num-images", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
