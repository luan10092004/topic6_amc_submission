# =========================================================
# IMPORTANT:
# Set thread-related environment variables BEFORE
# importing NumPy / PyTorch.
# =========================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import csv
import gc
import time
import random

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# 1. CONFIGURATION
# =========================================================

SEED = 42

NUM_SAMPLES = 1000

ROUNDS = 7

INITIAL_WARMUP = 100

ROUND_WARMUP = 50


torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


DEVICE = torch.device("cpu")


# =========================================================
# 2. OPTIONAL CPU AFFINITY
# =========================================================
# Pin this Python process to one logical CPU.
# This reduces scheduler/core migration noise.
# =========================================================

try:
    import psutil

    process = psutil.Process()

    available_cpus = process.cpu_affinity()

    if len(available_cpus) > 0:

        selected_cpu = available_cpus[0]

        process.cpu_affinity([
            selected_cpu
        ])

        print(
            "CPU affinity locked to logical CPU:",
            selected_cpu,
        )

    else:
        print(
            "CPU affinity list unavailable."
        )

except Exception as e:

    print(
        "CPU affinity not applied:",
        e,
    )


# =========================================================
# 3. PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "radio_ml_split.npz"
)

IMAGE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "images"
)

MODEL_DIR = (
    ROOT
    / "models"
)

SPEC_FILE = (
    IMAGE_DIR
    / "spectrograms.npy"
)

DENSITY_FILE = (
    IMAGE_DIR
    / "constellation_density.npy"
)

SUMMARY_CSV = (
    MODEL_DIR
    / "benchmark_inference_stable_results.csv"
)

ROUND_CSV = (
    MODEL_DIR
    / "benchmark_inference_stable_rounds.csv"
)


# =========================================================
# 4. MODEL DEFINITIONS
# =========================================================

class IQCNN1D(nn.Module):

    def __init__(
        self,
        num_classes=8,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                2,
                32,
                kernel_size=7,
                padding=3,
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        x = self.features(x)

        return self.classifier(x)


class CNN2D(nn.Module):

    def __init__(
        self,
        num_classes=8,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        x = self.features(x)

        return self.classifier(x)


# =========================================================
# 5. LOAD DATA
# =========================================================

print()
print(
    "Loading data..."
)

raw = np.load(
    DATA_FILE,
    allow_pickle=True,
)

X = np.asarray(
    raw["X"],
    dtype=np.float32,
)

test_idx = np.asarray(
    raw["test_idx"],
    dtype=np.int64,
)

class_names = [
    str(x)
    for x in raw["class_names"]
]

num_classes = len(
    class_names
)


rng = np.random.default_rng(
    SEED
)

positions = rng.choice(
    len(test_idx),
    size=NUM_SAMPLES,
    replace=False,
)

selected_idx = test_idx[
    positions
]


print(
    "Classes:",
    class_names,
)

print(
    "Benchmark samples:",
    len(selected_idx),
)


# =========================================================
# 6. LOAD REPRESENTATIONS
# =========================================================

spectrograms = np.load(
    SPEC_FILE,
    mmap_mode="r",
)

density_images = np.load(
    DENSITY_FILE,
    mmap_mode="r",
)


# =========================================================
# 7. PREPARE TENSORS
# EVERYTHING HERE IS OUTSIDE TIMER
# =========================================================

iq_tensor = torch.from_numpy(
    np.asarray(
        X[selected_idx],
        dtype=np.float32,
    )
)

spec_tensor = torch.from_numpy(
    np.asarray(
        spectrograms[
            selected_idx
        ],
        dtype=np.float32,
    )
).unsqueeze(1)

density_tensor = torch.from_numpy(
    np.asarray(
        density_images[
            selected_idx
        ],
        dtype=np.float32,
    )
).unsqueeze(1)


# Pre-create batch-size-1 tensor views.
# Even indexing / unsqueeze is therefore outside timer.

iq_samples = [
    iq_tensor[i:i+1]
    for i in range(NUM_SAMPLES)
]

spec_samples = [
    spec_tensor[i:i+1]
    for i in range(NUM_SAMPLES)
]

density_samples = [
    density_tensor[i:i+1]
    for i in range(NUM_SAMPLES)
]


print(
    "IQ input:",
    tuple(iq_tensor.shape),
)

print(
    "Spec input:",
    tuple(spec_tensor.shape),
)

print(
    "Density input:",
    tuple(density_tensor.shape),
)


# =========================================================
# 8. LOAD MODELS
# =========================================================

iq_model = IQCNN1D(
    num_classes
)

spec_model = CNN2D(
    num_classes
)

density_model = CNN2D(
    num_classes
)


iq_model.load_state_dict(
    torch.load(
        MODEL_DIR / "iq_cnn1d.pt",
        map_location="cpu",
        weights_only=True,
    )
)

spec_model.load_state_dict(
    torch.load(
        MODEL_DIR / "spectrogram_cnn2d.pt",
        map_location="cpu",
        weights_only=True,
    )
)

density_model.load_state_dict(
    torch.load(
        MODEL_DIR / "density_cnn2d.pt",
        map_location="cpu",
        weights_only=True,
    )
)


for model in [
    iq_model,
    spec_model,
    density_model,
]:

    model.to(
        DEVICE
    )

    model.eval()


# =========================================================
# 9. VERIFY PARAMETER COUNTS
# =========================================================

def count_parameters(
    model,
):

    return sum(
        p.numel()
        for p in model.parameters()
    )


print()
print(
    "Parameter verification:"
)

print(
    "IQ:",
    count_parameters(
        iq_model
    ),
)

print(
    "Spec:",
    count_parameters(
        spec_model
    ),
)

print(
    "Density:",
    count_parameters(
        density_model
    ),
)


# =========================================================
# 10. MODEL COLLECTION
# =========================================================

benchmark_items = {

    "IQ CNN 1D": {
        "model": iq_model,
        "samples": iq_samples,
    },

    "Spectrogram CNN 2D": {
        "model": spec_model,
        "samples": spec_samples,
    },

    "Constellation Density CNN 2D": {
        "model": density_model,
        "samples": density_samples,
    },
}


# =========================================================
# 11. INITIAL GLOBAL WARMUP
# =========================================================

print()
print(
    "Initial warm-up..."
)

with torch.inference_mode():

    for (
        model_name,
        item
    ) in benchmark_items.items():

        model = item[
            "model"
        ]

        samples = item[
            "samples"
        ]

        for i in range(
            INITIAL_WARMUP
        ):

            _ = model(
                samples[
                    i % NUM_SAMPLES
                ]
            )


print(
    "Warm-up complete."
)


# =========================================================
# 12. RUN ONE MODEL FOR ONE ROUND
# =========================================================

def run_round(
    model,
    samples,
):

    # ---------------------------------------------
    # Warm-up before this round
    # ---------------------------------------------

    with torch.inference_mode():

        for i in range(
            ROUND_WARMUP
        ):

            _ = model(
                samples[
                    i % NUM_SAMPLES
                ]
            )

    # ---------------------------------------------
    # Remove GC interruptions during timing
    # ---------------------------------------------

    gc.collect()
    gc.disable()

    latencies = []

    try:

        with torch.inference_mode():

            total_start = (
                time.perf_counter_ns()
            )

            for sample in samples:

                start = (
                    time.perf_counter_ns()
                )

                _ = model(
                    sample
                )

                end = (
                    time.perf_counter_ns()
                )

                latencies.append(
                    (
                        end - start
                    )
                    / 1_000_000.0
                )

            total_end = (
                time.perf_counter_ns()
            )

    finally:

        gc.enable()

    latencies = np.asarray(
        latencies,
        dtype=np.float64,
    )

    total_sec = (
        total_end
        - total_start
    ) / 1_000_000_000.0

    return {

        "median":
            float(
                np.median(
                    latencies
                )
            ),

        "mean":
            float(
                np.mean(
                    latencies
                )
            ),

        "p95":
            float(
                np.percentile(
                    latencies,
                    95,
                )
            ),

        "std":
            float(
                np.std(
                    latencies
                )
            ),

        "throughput":
            float(
                NUM_SAMPLES
                / total_sec
            ),
    }


# =========================================================
# 13. MULTI-ROUND BENCHMARK
# =========================================================

print()
print(
    "=" * 70
)

print(
    "STABLE MODEL-ONLY INFERENCE BENCHMARK"
)

print(
    "CPU | 1 thread | batch=1"
)

print(
    f"{ROUNDS} rounds Ã— "
    f"{NUM_SAMPLES} samples/model"
)

print(
    "Preprocessing excluded"
)

print(
    "=" * 70
)


round_results = {
    name: []
    for name in benchmark_items
}


order_rng = random.Random(
    SEED
)


for round_id in range(
    1,
    ROUNDS + 1,
):

    order = list(
        benchmark_items.keys()
    )

    order_rng.shuffle(
        order
    )

    print()
    print(
        f"ROUND {round_id}/{ROUNDS}"
    )

    print(
        "Order:",
        " -> ".join(order),
    )

    for model_name in order:

        item = benchmark_items[
            model_name
        ]

        result = run_round(
            item["model"],
            item["samples"],
        )

        result[
            "round"
        ] = round_id

        round_results[
            model_name
        ].append(
            result
        )

        print(
            f"{model_name:32s}"
            f" median={result['median']:.4f} ms"
            f" | mean={result['mean']:.4f}"
            f" | p95={result['p95']:.4f}"
            f" | throughput="
            f"{result['throughput']:.1f}/s"
        )


# =========================================================
# 14. SAVE PER-ROUND RESULTS
# =========================================================

round_rows = []

for model_name, results in round_results.items():

    for row in results:

        round_rows.append({

            "Model":
                model_name,

            "Round":
                row["round"],

            "Median Latency (ms/sample)":
                row["median"],

            "Mean Latency (ms/sample)":
                row["mean"],

            "P95 Latency (ms/sample)":
                row["p95"],

            "Std Latency (ms)":
                row["std"],

            "Measured Throughput (samples/s)":
                row["throughput"],
        })


with open(
    ROUND_CSV,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            round_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        round_rows
    )


# =========================================================
# 15. ROBUST FINAL SUMMARY
# =========================================================

summary_rows = []


print()
print(
    "=" * 70
)

print(
    "ROBUST FINAL SUMMARY"
)

print(
    "=" * 70
)


for model_name, results in round_results.items():

    medians = np.asarray([
        r["median"]
        for r in results
    ])

    means = np.asarray([
        r["mean"]
        for r in results
    ])

    p95s = np.asarray([
        r["p95"]
        for r in results
    ])

    throughputs = np.asarray([
        r["throughput"]
        for r in results
    ])


    robust_median = float(
        np.median(
            medians
        )
    )

    robust_mean = float(
        np.median(
            means
        )
    )

    robust_p95 = float(
        np.median(
            p95s
        )
    )

    robust_throughput = float(
        np.median(
            throughputs
        )
    )

    min_median = float(
        np.min(
            medians
        )
    )

    max_median = float(
        np.max(
            medians
        )
    )

    stability_ratio = (
        max_median
        / min_median
    )


    print()
    print(
        model_name
    )

    print(
        f"Round medians       : "
        f"{np.round(medians, 4).tolist()}"
    )

    print(
        f"FINAL median        : "
        f"{robust_median:.4f} ms/sample"
    )

    print(
        f"Median of means     : "
        f"{robust_mean:.4f} ms/sample"
    )

    print(
        f"Median of P95       : "
        f"{robust_p95:.4f} ms/sample"
    )

    print(
        f"Median throughput   : "
        f"{robust_throughput:.1f} samples/s"
    )

    print(
        f"Min round median    : "
        f"{min_median:.4f} ms"
    )

    print(
        f"Max round median    : "
        f"{max_median:.4f} ms"
    )

    print(
        f"Stability ratio     : "
        f"{stability_ratio:.3f}x"
    )


    summary_rows.append({

        "Model":
            model_name,

        "Rounds":
            ROUNDS,

        "Samples per Round":
            NUM_SAMPLES,

        "Robust Median Inference Latency (ms/sample)":
            robust_median,

        "Median of Round Means (ms/sample)":
            robust_mean,

        "Median of Round P95 (ms/sample)":
            robust_p95,

        "Median Measured Throughput (samples/s)":
            robust_throughput,

        "Min Round Median (ms/sample)":
            min_median,

        "Max Round Median (ms/sample)":
            max_median,

        "Stability Ratio":
            stability_ratio,

        "Parameters":
            count_parameters(
                benchmark_items[
                    model_name
                ]["model"]
            ),
    })


# =========================================================
# 16. SAVE FINAL SUMMARY
# =========================================================

with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


print()
print(
    "=" * 70
)

print(
    "STABLE BENCHMARK COMPLETE"
)

print(
    "=" * 70
)

print(
    "Summary CSV:"
)

print(
    SUMMARY_CSV
)

print()
print(
    "Per-round CSV:"
)

print(
    ROUND_CSV
)
