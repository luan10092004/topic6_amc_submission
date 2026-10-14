from pathlib import Path
import os
import csv

import psutil
import torch
import torch.nn as nn
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

DEVICE = torch.device("cpu")


class CNN1D(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN2D(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(model_class, filename):
    model = model_class()

    checkpoint = torch.load(
        MODEL_DIR / filename,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def tensor_memory_bytes(model):
    total = 0

    for p in model.parameters():
        total += p.numel() * p.element_size()

    for b in model.buffers():
        total += b.numel() * b.element_size()

    return total


def benchmark_memory(name, model, sample):
    process = psutil.Process(os.getpid())

    before = process.memory_info().rss

    with torch.inference_mode():
        for _ in range(50):
            model(sample)

    after = process.memory_info().rss

    model_bytes = tensor_memory_bytes(model)

    return {
        "Model": name,
        "Parameters": sum(p.numel() for p in model.parameters()),
        "Model tensor memory (MB)": model_bytes / (1024 ** 2),
        "Process RSS before inference (MB)": before / (1024 ** 2),
        "Process RSS after inference (MB)": after / (1024 ** 2),
        "RSS change (MB)": (after - before) / (1024 ** 2),
    }


models = [
    (
        "IQ CNN 1D",
        load_model(CNN1D, "iq_cnn1d.pt"),
        torch.zeros((1, 2, 128), dtype=torch.float32),
    ),
    (
        "Spectrogram CNN 2D",
        load_model(CNN2D, "spectrogram_cnn2d.pt"),
        torch.zeros((1, 1, 32, 32), dtype=torch.float32),
    ),
    (
        "Constellation Density CNN 2D",
        load_model(CNN2D, "density_cnn2d.pt"),
        torch.zeros((1, 1, 32, 32), dtype=torch.float32),
    ),
]


results = []

for name, model, sample in models:
    result = benchmark_memory(
        name,
        model,
        sample,
    )

    results.append(result)

    print(f"\n{name}")
    print(
        "Parameters:",
        f"{result['Parameters']:,}",
    )
    print(
        "Model tensor memory:",
        f"{result['Model tensor memory (MB)']:.5f} MB",
    )
    print(
        "RSS before:",
        f"{result['Process RSS before inference (MB)']:.3f} MB",
    )
    print(
        "RSS after:",
        f"{result['Process RSS after inference (MB)']:.3f} MB",
    )
    print(
        "RSS change:",
        f"{result['RSS change (MB)']:.3f} MB",
    )


output_path = MODEL_DIR / "benchmark_memory_results.csv"

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(results)


print(
    "\nSaved to:",
    output_path,
)

print(
    "Memory benchmark completed."
)

