from pathlib import Path

import numpy as np


# =========================================================
# CONFIGURATION
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

SPEC_FILE = (
    IMAGE_DIR
    / "spectrograms.npy"
)

DENSITY_FILE = (
    IMAGE_DIR
    / "constellation_density.npy"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "demo"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "demo_samples.npz"
)

SEED = 42

# 2 samples for each:
# 8 classes × 15 SNR values × 2 = 240 samples
SAMPLES_PER_CLASS_SNR = 2


# =========================================================
# LOAD FULL DATA
# =========================================================

print("Loading full dataset...")

raw = np.load(
    DATA_FILE,
    allow_pickle=True,
)

X = raw["X"]
y = raw["y"]
snrs = raw["snrs"]
test_idx = raw["test_idx"]
class_names = raw["class_names"]

spectrograms = np.load(
    SPEC_FILE,
    mmap_mode="r",
)

density_images = np.load(
    DENSITY_FILE,
    mmap_mode="r",
)


# =========================================================
# SELECT BALANCED DEMO SAMPLES
# =========================================================

rng = np.random.default_rng(
    SEED
)

selected_indices = []

unique_snrs = sorted(
    np.unique(
        snrs[test_idx]
    ).tolist()
)

num_classes = len(
    class_names
)


for class_id in range(
    num_classes
):

    for snr in unique_snrs:

        candidates = test_idx[
            (
                y[test_idx]
                == class_id
            )
            &
            (
                snrs[test_idx]
                == snr
            )
        ]

        if len(candidates) < SAMPLES_PER_CLASS_SNR:

            raise RuntimeError(
                f"Not enough samples for "
                f"class={class_names[class_id]}, "
                f"SNR={snr}"
            )

        chosen = rng.choice(
            candidates,
            size=SAMPLES_PER_CLASS_SNR,
            replace=False,
        )

        selected_indices.extend(
            chosen.tolist()
        )


selected_indices = np.asarray(
    selected_indices,
    dtype=np.int64,
)


# Shuffle while remaining deterministic
rng.shuffle(
    selected_indices
)


# =========================================================
# BUILD BUNDLE
# =========================================================

demo_X = np.asarray(
    X[selected_indices],
    dtype=np.float32,
)

demo_y = np.asarray(
    y[selected_indices],
)

demo_snrs = np.asarray(
    snrs[selected_indices],
)

demo_spec = np.asarray(
    spectrograms[selected_indices],
    dtype=np.float32,
)

demo_density = np.asarray(
    density_images[selected_indices],
    dtype=np.float32,
)


# =========================================================
# SAVE
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez_compressed(
    OUTPUT_FILE,

    X=demo_X,
    y=demo_y,
    snrs=demo_snrs,

    class_names=class_names,

    spectrograms=demo_spec,
    constellation_density=demo_density,

    # Keep original indices for traceability
    original_indices=selected_indices,
)


# =========================================================
# VERIFY
# =========================================================

print()
print("=" * 60)
print("DEMO BUNDLE CREATED")
print("=" * 60)

print(
    "Output:",
    OUTPUT_FILE,
)

print(
    "Samples:",
    len(demo_X),
)

print(
    "X shape:",
    demo_X.shape,
)

print(
    "Spectrogram shape:",
    demo_spec.shape,
)

print(
    "Density shape:",
    demo_density.shape,
)

print(
    "Classes:",
    [
        str(x)
        for x in class_names
    ],
)

print(
    "SNR values:",
    sorted(
        np.unique(
            demo_snrs
        ).tolist()
    ),
)

print()
print(
    "Samples per class / SNR:"
)

for class_id, class_name in enumerate(
    class_names
):

    counts = []

    for snr in unique_snrs:

        count = np.sum(
            (
                demo_y
                == class_id
            )
            &
            (
                demo_snrs
                == snr
            )
        )

        counts.append(
            int(count)
        )

    print(
        f"{str(class_name):8s}:",
        counts,
    )