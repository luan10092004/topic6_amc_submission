from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn as nn

from scipy.signal import stft
from scipy.ndimage import zoom


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

DEVICE = torch.device("cpu")

IMAGE_SIZE = 32

NUM_TEST_SAMPLES = 1000
WARMUP_RUNS = 50

torch.set_num_threads(1)

torch.manual_seed(42)
np.random.seed(42)


# =========================================================
# 2. MODELS
# =========================================================

class CNN1D(nn.Module):
    def __init__(self, num_classes=8):
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
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(
            self.features(x)
        )


class CNN2D(nn.Module):
    def __init__(self, num_classes=8):
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

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(
            self.features(x)
        )


# =========================================================
# 3. PREPROCESSING
# =========================================================

def make_spectrogram(iq):

    I = iq[0].astype(np.float32)
    Q = iq[1].astype(np.float32)

    signal = I + 1j * Q

    _, _, Z = stft(
        signal,
        fs=1.0,
        nperseg=32,
        noverlap=16,
        nfft=32,
        boundary=None,
        padded=False,
        return_onesided=False,
    )

    power = np.abs(Z) ** 2

    spec_db = 10.0 * np.log10(
        power + 1e-12
    )

    spec_db -= spec_db.min()

    if spec_db.max() > 0:
        spec_db /= spec_db.max()

    resized = zoom(
        spec_db,
        (
            IMAGE_SIZE / spec_db.shape[0],
            IMAGE_SIZE / spec_db.shape[1],
        ),
        order=1,
    )

    result = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32,
    )

    h = min(
        IMAGE_SIZE,
        resized.shape[0],
    )

    w = min(
        IMAGE_SIZE,
        resized.shape[1],
    )

    result[:h, :w] = resized[:h, :w]

    return result


def make_constellation_density(iq):

    I = iq[0].astype(np.float32)
    Q = iq[1].astype(np.float32)

    amplitude = np.sqrt(
        I ** 2 + Q ** 2
    )

    scale = amplitude.max()

    if scale > 1e-12:
        I /= scale
        Q /= scale

    hist, _, _ = np.histogram2d(
        I,
        Q,
        bins=IMAGE_SIZE,
        range=[
            [-1, 1],
            [-1, 1],
        ],
    )

    hist = np.log1p(hist)

    if hist.max() > 0:
        hist /= hist.max()

    return hist.T.astype(
        np.float32
    )


# =========================================================
# 4. LOAD MODELS
# =========================================================

def load_model(
    model_class,
    filename,
):

    path = MODEL_DIR / filename

    model = model_class()

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model


iq_model = load_model(
    CNN1D,
    "iq_cnn1d.pt",
)

spec_model = load_model(
    CNN2D,
    "spectrogram_cnn2d.pt",
)

density_model = load_model(
    CNN2D,
    "density_cnn2d.pt",
)


# =========================================================
# 5. LOAD RAW TEST DATA
# =========================================================

raw = np.load(
    DATA_DIR / "radio_ml_split.npz"
)

X = raw["X"]
test_idx = raw["test_idx"]

rng = np.random.default_rng(
    42
)

number_to_use = min(
    NUM_TEST_SAMPLES,
    len(test_idx),
)

selected_positions = rng.choice(
    len(test_idx),
    size=number_to_use,
    replace=False,
)

selected_indices = test_idx[
    selected_positions
]

print(
    "Device:",
    DEVICE,
)

print(
    "Number of test bursts:",
    number_to_use,
)


# =========================================================
# 6. END-TO-END FUNCTIONS
# =========================================================

def run_iq_pipeline(iq):

    tensor = torch.from_numpy(
        np.array(
            iq[None, :, :],
            dtype=np.float32,
            copy=True,
        )
    )

    with torch.inference_mode():

        output = iq_model(
            tensor
        )

        prediction = torch.argmax(
            output,
            dim=1,
        ).item()

    return prediction


def run_spectrogram_pipeline(iq):

    image = make_spectrogram(
        iq
    )

    tensor = torch.from_numpy(
        image[
            None,
            None,
            :,
            :
        ]
    )

    with torch.inference_mode():

        output = spec_model(
            tensor
        )

        prediction = torch.argmax(
            output,
            dim=1,
        ).item()

    return prediction


def run_density_pipeline(iq):

    image = make_constellation_density(
        iq
    )

    tensor = torch.from_numpy(
        image[
            None,
            None,
            :,
            :
        ]
    )

    with torch.inference_mode():

        output = density_model(
            tensor
        )

        prediction = torch.argmax(
            output,
            dim=1,
        ).item()

    return prediction


# =========================================================
# 7. BENCHMARK FUNCTION
# =========================================================

def benchmark_pipeline(
    name,
    function,
):

    print(
        f"\nBenchmarking {name}..."
    )

    warmup_iq = np.array(
        X[selected_indices[0]],
        dtype=np.float32,
        copy=True,
    )

    # Warm-up
    for _ in range(
        WARMUP_RUNS
    ):
        function(
            warmup_iq
        )

    times_ms = []

    for sample_index in selected_indices:

        iq = np.array(
            X[sample_index],
            dtype=np.float32,
            copy=True,
        )

        start = time.perf_counter_ns()

        function(
            iq
        )

        end = time.perf_counter_ns()

        elapsed_ms = (
            end - start
        ) / 1_000_000.0

        times_ms.append(
            elapsed_ms
        )

    times_ms = np.asarray(
        times_ms,
        dtype=np.float64,
    )

    result = {
        "Model": name,

        "Samples": len(times_ms),

        "Median E2E latency (ms/sample)":
            float(
                np.median(times_ms)
            ),

        "Mean E2E latency (ms/sample)":
            float(
                np.mean(times_ms)
            ),

        "P95 E2E latency (ms/sample)":
            float(
                np.percentile(
                    times_ms,
                    95,
                )
            ),

        "Minimum E2E latency (ms/sample)":
            float(
                np.min(times_ms)
            ),
    }

    result["Throughput (samples/s)"] = (
        1000.0
        / result[
            "Median E2E latency (ms/sample)"
        ]
    )

    return result


# =========================================================
# 8. RUN
# =========================================================

results = []

results.append(
    benchmark_pipeline(
        "IQ CNN 1D",
        run_iq_pipeline,
    )
)

results.append(
    benchmark_pipeline(
        "Spectrogram CNN 2D",
        run_spectrogram_pipeline,
    )
)

results.append(
    benchmark_pipeline(
        "Constellation Density CNN 2D",
        run_density_pipeline,
    )
)


# =========================================================
# 9. PRINT RESULTS
# =========================================================

print(
    "\n===== END-TO-END BENCHMARK ====="
)

for result in results:

    print(
        f"\n{result['Model']}"
    )

    print(
        "Median:",
        f"{result['Median E2E latency (ms/sample)']:.4f}",
        "ms/sample",
    )

    print(
        "Mean:",
        f"{result['Mean E2E latency (ms/sample)']:.4f}",
        "ms/sample",
    )

    print(
        "P95:",
        f"{result['P95 E2E latency (ms/sample)']:.4f}",
        "ms/sample",
    )

    print(
        "Throughput:",
        f"{result['Throughput (samples/s)']:.1f}",
        "samples/s",
    )


# =========================================================
# 10. SAVE CSV
# =========================================================

output_path = (
    MODEL_DIR
    / "benchmark_end_to_end_results.csv"
)

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(
        results
    )


print(
    "\nSaved to:",
    output_path,
)

print(
    "End-to-end benchmark completed."
)
