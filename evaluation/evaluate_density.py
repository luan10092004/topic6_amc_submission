
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")
IMAGE_DIR = DATA_DIR / "images"

PRED_PATH = MODEL_DIR / "density_test_predictions.npz"
META_PATH = IMAGE_DIR / "metadata.npz"

OUTPUT_DIR = MODEL_DIR / "evaluation_density"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Dang doc ket qua du doan...")

    pred_data = np.load(PRED_PATH, allow_pickle=False)
    metadata = np.load(META_PATH, allow_pickle=False)

    y_true = pred_data["y_true"]
    y_pred = pred_data["y_pred"]
    test_idx = pred_data["test_idx"]
    class_names = pred_data["class_names"].tolist()

    # SNR cua tung mau trong dataset goc
    snrs = metadata["snrs"][test_idx]

    # ======================================
    # 1. METRICS TONG QUAN
    # ======================================
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\n===== TONG QUAN TEST =====")
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro-F1 : {macro_f1:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    # ======================================
    # 2. CONFUSION MATRIX TOAN BO TEST
    # ======================================
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=45,
        colorbar=False,
    )

    ax.set_title("2D CNN Density - Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrix_all.png",
        dpi=200,
    )
    plt.close(fig)

    # ======================================
    # 3. ACCURACY THEO SNR
    # ======================================
    unique_snrs = np.sort(np.unique(snrs))
    snr_accuracies = []

    print("\n===== ACCURACY THEO SNR =====")

    for snr in unique_snrs:
        mask = snrs == snr

        snr_acc = accuracy_score(
            y_true[mask],
            y_pred[mask],
        )

        snr_accuracies.append(snr_acc)

        print(
            f"SNR {int(snr):>3} dB: "
            f"Accuracy = {snr_acc * 100:.2f}% "
            f"(N={mask.sum()})"
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        unique_snrs,
        np.array(snr_accuracies) * 100,
        marker="o",
    )

    ax.set_title("2D CNN Density - Accuracy vs SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(unique_snrs)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "accuracy_vs_snr.png",
        dpi=200,
    )
    plt.close(fig)

    # ======================================
    # 4. CONFUSION MATRIX TAI -4, +4, +14 dB
    # ======================================
    target_snrs = [-4, 4, 14]

    for target_snr in target_snrs:
        mask = snrs == target_snr

        if not np.any(mask):
            print(
                f"Khong co mau test tai SNR {target_snr} dB"
            )
            continue

        cm_snr = confusion_matrix(
            y_true[mask],
            y_pred[mask],
            labels=np.arange(len(class_names)),
            normalize="true",
        )

        fig, ax = plt.subplots(figsize=(9, 7))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_snr,
            display_labels=class_names,
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            values_format=".2f",
            xticks_rotation=45,
            colorbar=False,
        )

        ax.set_title(
            f"2D CNN Density - Confusion Matrix at {target_snr} dB"
        )

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"confusion_matrix_snr_{target_snr}.png",
            dpi=200,
        )
        plt.close(fig)

        snr_f1 = f1_score(
            y_true[mask],
            y_pred[mask],
            average="macro",
            zero_division=0,
        )

        print(
            f"SNR {target_snr:+} dB: "
            f"Accuracy={accuracy_score(y_true[mask], y_pred[mask]):.4f}, "
            f"Macro-F1={snr_f1:.4f}, "
            f"N={mask.sum()}"
        )

    print("\nDa luu bieu do tai:", OUTPUT_DIR.resolve())
    print("Hoan tat danh gia!")


if __name__ == "__main__":
    main()
