from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================
# 1. PATHS
# =========================================
MODEL_FILES = {
    "Spectrogram 2D CNN":
        Path("models/spectrogram_test_predictions.npz"),

    "Constellation Density 2D CNN":
        Path("models/density_test_predictions.npz"),

    "I/Q 1D CNN":
        Path("models/iq_cnn1d_test_predictions.npz"),
}

META_PATH = Path("data/processed/images/metadata.npz")
OUT_DIR = Path("models/confusion_by_snr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SNRS = [-4, 4, 14]

# =========================================
# 2. LOAD METADATA
# =========================================
metadata = np.load(META_PATH, allow_pickle=True)

snrs_all = metadata["snrs"]
class_names = metadata["class_names"].astype(str).tolist()

print("Classes:", class_names)

# =========================================
# 3. LOAD PREDICTIONS
# =========================================
models_data = {}

for model_name, path in MODEL_FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"KhÃ´ng tÃ¬m tháº¥y file: {path}"
        )

    data = np.load(path, allow_pickle=True)

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    test_idx = data["test_idx"]

    snrs_test = snrs_all[test_idx]

    models_data[model_name] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "snrs": snrs_test,
    }

    print(f"\nLoaded: {model_name}")
    print("Number of predictions:", len(y_true))

# =========================================
# 4. CONFUSION MATRIX BY SNR
# =========================================
labels = np.arange(len(class_names))

for snr in TARGET_SNRS:
    print(f"\n===== SNR = {snr} dB =====")

    fig, axes = plt.subplots(
        1, 3, figsize=(21, 6), constrained_layout=True
    )

    for ax, (model_name, data) in zip(
        axes, models_data.items()
    ):
        mask = data["snrs"] == snr

        if not np.any(mask):
            ax.set_title(f"{model_name}\nNo samples")
            ax.axis("off")
            continue

        y_true_snr = data["y_true"][mask]
        y_pred_snr = data["y_pred"][mask]

        cm = confusion_matrix(
            y_true_snr,
            y_pred_snr,
            labels=labels,
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names,
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d",
            xticks_rotation=60,
        )

        accuracy = np.mean(y_true_snr == y_pred_snr)

        ax.set_title(
            f"{model_name}\n"
            f"SNR = {snr} dB | Accuracy = {accuracy:.2%}"
        )
        ax.tick_params(axis="both", labelsize=8)

        print(
            f"{model_name}: "
            f"Accuracy = {accuracy:.4f} "
            f"({accuracy:.2%}), "
            f"samples = {len(y_true_snr)}"
        )

    fig.suptitle(
        f"Confusion Matrix Comparison at SNR = {snr} dB",
        fontsize=15,
    )

    output_path = OUT_DIR / f"confusion_snr_{snr:+d}dB.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)

print("\nAll confusion matrices generated.")
print("Output folder:", OUT_DIR)
