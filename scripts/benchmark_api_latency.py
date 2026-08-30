"""Benchmark inference latency against a running API server."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def load_image_bytes() -> bytes:
    manifest_path = PROJECT_ROOT / "data" / "plantvillage" / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        row = json.load(handle)["test"][0]
    with open(row["image_path"], "rb") as handle:
        return handle.read()


def benchmark(base_url: str, runs: int) -> dict[str, float | int]:
    image_bytes = load_image_bytes()
    latencies: list[float] = []
    client = httpx.Client(base_url=base_url, timeout=30.0)

    for _ in range(3):
        client.post("/api/v1/predict", files={"image": ("leaf.jpg", image_bytes, "image/jpeg")})

    for _ in range(runs):
        start = time.perf_counter()
        response = client.post("/api/v1/predict", files={"image": ("leaf.jpg", image_bytes, "image/jpeg")})
        response.raise_for_status()
        payload = response.json()
        total_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(total_ms)
        _ = payload["inference_ms"]

    summary = {
        "base_url": base_url,
        "runs": runs,
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }
    output_path = RESULTS_DIR / "docker_latency_benchmark.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "all_ms": [round(v, 2) for v in latencies]}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7861")
    parser.add_argument("--runs", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(**vars(parse_args()))
