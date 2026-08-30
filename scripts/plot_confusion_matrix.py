"""Generate a confusion matrix plot for the README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    with (RESULTS_DIR / "evaluation_metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    matrix = np.array(metrics["confusion_matrix"])
    labels = [name.replace("___", "\n") for name in metrics["class_names"]]

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="Greens", fmt="d")
    plt.title("Plant Disease Classifier — Confusion Matrix (Test Split)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    output_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
