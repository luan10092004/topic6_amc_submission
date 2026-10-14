from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


# =========================================================
# 1. PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

FIGURE_DIR = (
    ROOT
    / "results"
    / "final_figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

SNR_LEVELS = np.array([
    -10,
    -8,
    -6,
    -4,
    -2,
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
])


MODEL_FILES = {
    "IQ CNN 1D":
        MODEL_DIR
        / "iq_cnn1d_test_predictions.npz",

    "Spectrogram CNN 2D":
        MODEL_DIR
        / "spectrogram_test_predictions.npz",

    "Constellation Density CNN 2D":
        MODEL_DIR
        / "density_test_predictions.npz",
}


# =========================================================
# 3. LOAD ORIGINAL DATASET
# =========================================================

raw = np.load(
    DATA_DIR / "radio_ml_split.npz",
    allow_pickle=True,
)

global_y = np.asarray(
    raw["y"],
    dtype=np.int64,
)

global_snrs = np.asarray(
    raw["snrs"],
)

class_names = [
    str(x)
    for x in raw["class_names"]
]

num_classes = len(class_names)

print(
    "Classes:",
    class_names,
)

print(
    "SNR levels:",
    SNR_LEVELS.tolist(),
)


# =========================================================
# 4. COMPUTE MACRO-F1 FOR EACH MODEL / SNR
# =========================================================

results = {}

csv_rows = []

for model_name, path in MODEL_FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find prediction file: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True,
    )

    y_true = np.asarray(
        data["y_true"],
        dtype=np.int64,
    )

    y_pred = np.asarray(
        data["y_pred"],
        dtype=np.int64,
    )

    test_idx = np.asarray(
        data["test_idx"],
        dtype=np.int64,
    )

    # SNR values corresponding to the exact
    # samples stored in this prediction file
    snrs = global_snrs[
        test_idx
    ]

    # Safety verification
    original_y = global_y[
        test_idx
    ]

    if not np.array_equal(
        y_true,
        original_y,
    ):
        raise ValueError(
            f"Label mismatch for {model_name}"
        )

    model_f1 = []

    print()
    print(
        "================================"
    )
    print(
        model_name
    )
    print(
        "================================"
    )

    for snr in SNR_LEVELS:

        mask = (
            snrs == snr
        )

        y_true_snr = y_true[
            mask
        ]

        y_pred_snr = y_pred[
            mask
        ]

        if len(y_true_snr) == 0:
            raise ValueError(
                f"No samples found for "
                f"{model_name} at SNR {snr}"
            )

        macro_f1 = f1_score(
            y_true_snr,
            y_pred_snr,
            labels=np.arange(
                num_classes
            ),
            average="macro",
            zero_division=0,
        )

        model_f1.append(
            macro_f1
        )

        csv_rows.append({
            "Model": model_name,
            "SNR (dB)": int(snr),
            "Macro-F1": float(macro_f1),
            "Macro-F1 (%)": float(
                macro_f1 * 100
            ),
            "Samples": int(
                len(y_true_snr)
            ),
        })

        print(
            f"SNR {snr:+3d} dB | "
            f"Samples {len(y_true_snr):4d} | "
            f"Macro-F1 "
            f"{macro_f1 * 100:6.2f}%"
        )

    results[
        model_name
    ] = np.asarray(
        model_f1,
        dtype=np.float64,
    )


# =========================================================
# 5. SAVE NUMERIC RESULTS
# =========================================================

csv_path = (
    MODEL_DIR
    / "macro_f1_by_snr.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Model",
            "SNR (dB)",
            "Macro-F1",
            "Macro-F1 (%)",
            "Samples",
        ],
    )

    writer.writeheader()
    writer.writerows(
        csv_rows
    )

print()
print(
    "Saved CSV:",
    csv_path,
)


# =========================================================
# 6. PLOT MACRO-F1 VS SNR
# =========================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

markers = {
    "IQ CNN 1D": "o",
    "Spectrogram CNN 2D": "s",
    "Constellation Density CNN 2D": "^",
}

for model_name, values in results.items():

    ax.plot(
        SNR_LEVELS,
        values * 100,
        marker=markers[
            model_name
        ],
        linewidth=2,
        markersize=6,
        label=model_name,
    )


ax.set_xlabel(
    "SNR (dB)"
)

ax.set_ylabel(
    "Macro-F1 (%)"
)

ax.set_title(
    "Macro-F1 vs SNR"
)

ax.set_xticks(
    SNR_LEVELS
)

ax.set_ylim(
    0,
    100,
)

ax.grid(
    alpha=0.3
)

ax.legend()

fig.tight_layout()


figure_path = (
    FIGURE_DIR
    / "07_macro_f1_vs_snr.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    "Saved figure:",
    figure_path,
)


# =========================================================
# 7. CHECK KEY SNR VALUES
# =========================================================

print()
print(
    "===== CHECK AT KEY SNR POINTS ====="
)

for snr in [
    -4,
    4,
    14,
]:

    position = int(
        np.where(
            SNR_LEVELS == snr
        )[0][0]
    )

    print()
    print(
        f"SNR = {snr:+d} dB"
    )

    for model_name in MODEL_FILES:

        value = (
            results[
                model_name
            ][position]
            * 100
        )

        print(
            f"{model_name:32s}: "
            f"{value:.2f}%"
        )


print()
print(
    "===================================="
)

print(
    "MACRO-F1 VS SNR COMPLETED"
)

print(
    "===================================="
)
