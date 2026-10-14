from pathlib import Path
import json
import csv

import matplotlib.pyplot as plt


# =========================================================
# 1. PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "results" / "final_figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. TRAINING HISTORY FILES
# =========================================================

HISTORY_FILES = {
    "IQ CNN 1D":
        MODEL_DIR / "iq_cnn1d_training_history.json",

    "Spectrogram CNN 2D":
        MODEL_DIR / "spectrogram_training_history.json",

    "Constellation Density CNN 2D":
        MODEL_DIR / "density_training_history.json",
}


# =========================================================
# 3. LOAD + VERIFY HISTORIES
# =========================================================

histories = {}

required_keys = {
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
}


for model_name, path in HISTORY_FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        history = json.load(f)

    if not history:
        raise ValueError(
            f"Empty history file: {path}"
        )

    missing = (
        required_keys
        - set(history[0].keys())
    )

    if missing:
        raise KeyError(
            f"{path.name} missing keys: {missing}"
        )

    histories[model_name] = history

    print()
    print("=" * 55)
    print(model_name)
    print("=" * 55)

    print(
        "History file:",
        path.name,
    )

    print(
        "Epochs saved:",
        len(history),
    )

    print(
        "Available keys:",
        list(history[0].keys()),
    )

    # Best epoch according to validation accuracy
    best_index = max(
        range(len(history)),
        key=lambda i: history[i]["val_acc"],
    )

    best = history[best_index]

    epoch_value = best.get(
        "epoch",
        best_index + 1,
    )

    print(
        "Best epoch:",
        epoch_value,
    )

    print(
        "Best validation accuracy:",
        f"{best['val_acc'] * 100:.2f}%"
    )

    print(
        "Validation loss at best epoch:",
        f"{best['val_loss']:.6f}"
    )


# =========================================================
# 4. SAVE COMBINED HISTORY CSV
# =========================================================

csv_path = (
    MODEL_DIR
    / "training_history_combined.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Epoch",
        "Train Loss",
        "Validation Loss",
        "Train Accuracy",
        "Validation Accuracy",
    ])

    for model_name, history in histories.items():

        for i, row in enumerate(history):

            epoch = row.get(
                "epoch",
                i + 1,
            )

            writer.writerow([
                model_name,
                epoch,
                row["train_loss"],
                row["val_loss"],
                row["train_acc"],
                row["val_acc"],
            ])

print()
print(
    "Saved combined CSV:",
    csv_path,
)


# =========================================================
# 5. FIGURE: TRAINING / VALIDATION LOSS
# =========================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.5),
)

for ax, (
    model_name,
    history,
) in zip(
    axes,
    histories.items(),
):

    epochs = [
        row.get("epoch", i + 1)
        for i, row in enumerate(history)
    ]

    train_loss = [
        row["train_loss"]
        for row in history
    ]

    val_loss = [
        row["val_loss"]
        for row in history
    ]

    best_index = max(
        range(len(history)),
        key=lambda i: history[i]["val_acc"],
    )

    best_epoch = epochs[
        best_index
    ]

    ax.plot(
        epochs,
        train_loss,
        marker="o",
        markersize=4,
        label="Training Loss",
    )

    ax.plot(
        epochs,
        val_loss,
        marker="s",
        markersize=4,
        label="Validation Loss",
    )

    ax.axvline(
        best_epoch,
        linestyle="--",
        linewidth=1.5,
        label=f"Best Epoch = {best_epoch}",
    )

    ax.set_title(
        model_name
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Cross-Entropy Loss"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        fontsize=8
    )


fig.suptitle(
    "Training and Validation Loss Curves",
    fontsize=16,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

loss_path = (
    OUTPUT_DIR
    / "08_training_validation_loss.png"
)

fig.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "Saved loss figure:",
    loss_path,
)


# =========================================================
# 6. FIGURE: TRAINING / VALIDATION ACCURACY
# =========================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.5),
)

for ax, (
    model_name,
    history,
) in zip(
    axes,
    histories.items(),
):

    epochs = [
        row.get("epoch", i + 1)
        for i, row in enumerate(history)
    ]

    train_acc = [
        row["train_acc"] * 100
        for row in history
    ]

    val_acc = [
        row["val_acc"] * 100
        for row in history
    ]

    best_index = max(
        range(len(history)),
        key=lambda i: history[i]["val_acc"],
    )

    best_epoch = epochs[
        best_index
    ]

    ax.plot(
        epochs,
        train_acc,
        marker="o",
        markersize=4,
        label="Training Accuracy",
    )

    ax.plot(
        epochs,
        val_acc,
        marker="s",
        markersize=4,
        label="Validation Accuracy",
    )

    ax.axvline(
        best_epoch,
        linestyle="--",
        linewidth=1.5,
        label=f"Best Epoch = {best_epoch}",
    )

    ax.set_title(
        model_name
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        fontsize=8
    )


fig.suptitle(
    "Training and Validation Accuracy Curves",
    fontsize=16,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

accuracy_path = (
    OUTPUT_DIR
    / "09_training_validation_accuracy.png"
)

fig.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "Saved accuracy figure:",
    accuracy_path,
)


# =========================================================
# 7. PRINT FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print("TRAINING CURVES COMPLETE")
print("=" * 60)

for model_name, history in histories.items():

    best_index = max(
        range(len(history)),
        key=lambda i: history[i]["val_acc"],
    )

    best = history[
        best_index
    ]

    best_epoch = best.get(
        "epoch",
        best_index + 1,
    )

    print(
        f"{model_name:32s}"
        f" | epochs={len(history):2d}"
        f" | best epoch={best_epoch:2d}"
        f" | best val acc="
        f"{best['val_acc'] * 100:.2f}%"
    )

print()
print(
    "Outputs:"
)

print(
    loss_path
)

print(
    accuracy_path
)

print(
    csv_path
)
