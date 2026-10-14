from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# =========================================================
# 1. PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = (
    ROOT
    / "results"
    / "final_figures"
    / "normalized_confusions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

TARGET_SNRS = [
    -4,
    4,
    14,
]


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
# 3. LOAD ORIGINAL DATASET METADATA
# =========================================================

raw = np.load(
    DATA_DIR / "radio_ml_split.npz",
    allow_pickle=True,
)

required_raw_keys = {
    "y",
    "snrs",
    "test_idx",
    "class_names",
}

missing = (
    required_raw_keys
    - set(raw.files)
)

if missing:
    raise KeyError(
        f"Missing keys in radio_ml_split.npz: {missing}"
    )


global_y = raw["y"]
global_snrs = raw["snrs"]

class_names = [
    str(x)
    for x in raw["class_names"]
]

num_classes = len(
    class_names
)

print(
    "Class names:",
    class_names,
)

print(
    "Number of classes:",
    num_classes,
)


# =========================================================
# 4. LOAD AND VERIFY PREDICTIONS
# =========================================================

prediction_data = {}

for model_name, path in MODEL_FILES.items():

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True,
    )

    required_keys = {
        "y_true",
        "y_pred",
        "test_idx",
    }

    missing = (
        required_keys
        - set(data.files)
    )

    if missing:

        raise KeyError(
            f"{path.name} is missing keys: "
            f"{missing}"
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

    if not (
        len(y_true)
        == len(y_pred)
        == len(test_idx)
    ):

        raise ValueError(
            f"Length mismatch in {path.name}"
        )

    # -----------------------------------------------------
    # Derive SNR from the original dataset using
    # the exact test indices stored in the prediction file.
    # -----------------------------------------------------

    snrs = np.asarray(
        global_snrs[test_idx]
    )

    original_labels = np.asarray(
        global_y[test_idx],
        dtype=np.int64,
    )

    # -----------------------------------------------------
    # Verify stored labels match original dataset labels.
    # -----------------------------------------------------

    labels_match = np.array_equal(
        y_true,
        original_labels,
    )

    if not labels_match:

        raise ValueError(
            f"y_true in {path.name} does not "
            "match the original dataset labels."
        )

    prediction_data[
        model_name
    ] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "test_idx": test_idx,
        "snrs": snrs,
    }

    print()
    print(
        "Loaded:",
        model_name,
    )

    print(
        "Prediction file:",
        path.name,
    )

    print(
        "Samples:",
        len(y_true),
    )

    print(
        "Dataset label verification:",
        "PASS",
    )


# =========================================================
# 5. FUNCTION: NORMALIZED CONFUSION MATRIX
# =========================================================

def get_normalized_confusion(
    y_true,
    y_pred,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            num_classes
        ),
        normalize="true",
    )

    return cm


# =========================================================
# 6. FUNCTION: DRAW ONE MATRIX
# =========================================================

def draw_matrix(
    ax,
    cm,
    title,
):

    image = ax.imshow(
        cm,
        vmin=0,
        vmax=1,
        cmap="Blues",
    )

    ax.set_title(
        title,
        fontsize=11,
    )

    ax.set_xlabel(
        "Predicted modulation"
    )

    ax.set_ylabel(
        "True modulation"
    )

    ax.set_xticks(
        np.arange(
            num_classes
        )
    )

    ax.set_yticks(
        np.arange(
            num_classes
        )
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        class_names,
    )

    # -----------------------------------------------------
    # Write percentages inside cells.
    # -----------------------------------------------------

    for i in range(
        num_classes
    ):

        for j in range(
            num_classes
        ):

            value = cm[i, j]

            text_color = (
                "white"
                if value > 0.5
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value * 100:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    return image


# =========================================================
# 7. GENERATE 9 INDIVIDUAL MATRICES
# =========================================================

print()
print(
    "========================================"
)
print(
    "GENERATING INDIVIDUAL CONFUSION MATRICES"
)
print(
    "========================================"
)


for model_name, data in prediction_data.items():

    for snr in TARGET_SNRS:

        mask = (
            data["snrs"]
            == snr
        )

        y_true_snr = (
            data["y_true"][mask]
        )

        y_pred_snr = (
            data["y_pred"][mask]
        )

        if len(
            y_true_snr
        ) == 0:

            raise ValueError(
                f"No samples found for "
                f"{model_name}, SNR={snr}"
            )

        cm = get_normalized_confusion(
            y_true_snr,
            y_pred_snr,
        )

        # -------------------------------------------------
        # Verification:
        # Each non-empty row should sum to approximately 1.
        # -------------------------------------------------

        row_sums = cm.sum(
            axis=1
        )

        print()
        print(
            f"{model_name} | "
            f"SNR = {snr:+d} dB"
        )

        print(
            "Number of samples:",
            len(y_true_snr),
        )

        print(
            "Row sums:",
            np.round(
                row_sums,
                6,
            ),
        )

        print(
            "Accuracy at this SNR:",
            f"{np.mean(y_true_snr == y_pred_snr) * 100:.2f}%"
        )

        # -------------------------------------------------
        # Draw
        # -------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(8, 7)
        )

        image = draw_matrix(
            ax,
            cm,
            (
                f"{model_name}\n"
                f"Row-Normalized Confusion Matrix "
                f"at SNR = {snr:+d} dB"
            ),
        )

        cbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        cbar.set_label(
            "Fraction of true class"
        )

        fig.tight_layout()

        safe_model_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        if snr >= 0:
            snr_text = (
                f"plus_{snr}"
            )
        else:
            snr_text = (
                f"minus_{abs(snr)}"
            )

        output_path = (
            OUTPUT_DIR
            / (
                f"{safe_model_name}"
                f"_snr_{snr_text}.png"
            )
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        print(
            "Saved:",
            output_path,
        )


# =========================================================
# 8. GENERATE 3 COMBINED FIGURES
# One figure per SNR, containing all 3 models
# =========================================================

print()
print(
    "========================================"
)
print(
    "GENERATING COMBINED CONFUSION MATRICES"
)
print(
    "========================================"
)


for snr in TARGET_SNRS:

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 6),
    )

    last_image = None

    for ax, (
        model_name,
        data,
    ) in zip(
        axes,
        prediction_data.items(),
    ):

        mask = (
            data["snrs"]
            == snr
        )

        y_true_snr = (
            data["y_true"][mask]
        )

        y_pred_snr = (
            data["y_pred"][mask]
        )

        cm = get_normalized_confusion(
            y_true_snr,
            y_pred_snr,
        )

        accuracy = (
            np.mean(
                y_true_snr
                == y_pred_snr
            )
            * 100
        )

        last_image = draw_matrix(
            ax,
            cm,
            (
                f"{model_name}\n"
                f"Accuracy = {accuracy:.2f}%"
            ),
        )

    fig.suptitle(
        (
            "Row-Normalized Confusion Matrices "
            f"at SNR = {snr:+d} dB"
        ),
        fontsize=16,
    )

    # -----------------------------------------------------
    # Shared colorbar
    # -----------------------------------------------------

    cbar = fig.colorbar(
        last_image,
        ax=axes.ravel().tolist(),
        fraction=0.018,
        pad=0.02,
    )

    cbar.set_label(
        "Fraction of true class"
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.92,
        bottom=0.18,
        top=0.82,
        wspace=0.35,
    )

    if snr >= 0:
        snr_text = (
            f"plus_{snr}"
        )
    else:
        snr_text = (
            f"minus_{abs(snr)}"
        )

    output_path = (
        OUTPUT_DIR
        / (
            f"combined_normalized_"
            f"confusion_snr_{snr_text}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        output_path,
    )


# =========================================================
# 9. FINISHED
# =========================================================

print()
print(
    "========================================"
)
print(
    "NORMALIZED CONFUSION MATRICES COMPLETE"
)
print(
    "========================================"
)

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Expected output:"
)

print(
    "9 individual matrices"
)

print(
    "+"
)

print(
    "3 combined comparison figures"
)
