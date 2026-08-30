"""Download and prepare PlantVillage images for fine-tuning."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "plantvillage"
FOCUS_CROPS = {"Tomato", "Potato", "Apple", "Pepper_bell"}


def parse_label_from_path(path: str) -> str:
    """Extract class label from a PlantVillage relative path."""
    parts = Path(path).parts
    for part in parts:
        if "___" in part:
            return part
    raise ValueError(f"Could not parse label from path: {path}")


def parse_crop(label: str) -> str:
    return label.split("___", 1)[0]


def download_and_extract() -> Path:
    """Download PlantVillage archive from Hugging Face and extract it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = Path(
        hf_hub_download(
            repo_id="mohanty/PlantVillage",
            filename="data.zip",
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
    )
    extract_root = DATA_DIR / "extracted"
    if not extract_root.exists():
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_root)
    return extract_root


def load_split_rows(split_name: str, extract_root: Path, max_per_class: int) -> list[dict]:
    """Load image paths for a split, filtered to focus crops."""
    split_file = hf_hub_download(
        repo_id="mohanty/PlantVillage",
        filename=f"splits/color_{split_name}.txt",
        repo_type="dataset",
    )
    per_class: dict[str, int] = defaultdict(int)
    rows: list[dict] = []

    with open(split_file, encoding="utf-8") as handle:
        for line in handle:
            rel_path = line.strip()
            if not rel_path:
                continue
            label = parse_label_from_path(rel_path)
            crop = parse_crop(label)
            if crop not in FOCUS_CROPS:
                continue
            if per_class[label] >= max_per_class:
                continue
            image_path = extract_root / rel_path
            if not image_path.exists():
                # Some archives nest under raw/
                alt_path = extract_root / "raw" / rel_path
                if alt_path.exists():
                    image_path = alt_path
                else:
                    continue
            per_class[label] += 1
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "crop": crop,
                    "disease": label.split("___", 1)[1],
                }
            )
    return rows


def write_manifest(max_per_class: int) -> dict[str, list[dict]]:
    """Create train/test manifests with local image paths."""
    extract_root = download_and_extract()
    manifest = {
        "train": load_split_rows("train", extract_root, max_per_class),
        "test": load_split_rows("test", extract_root, max_per_class),
    }
    manifest_path = DATA_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        f"Prepared {len(manifest['train'])} train and {len(manifest['test'])} test images "
        f"across {len({row['label'] for row in manifest['train']})} classes."
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-class", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    write_manifest(parse_args().max_per_class)
