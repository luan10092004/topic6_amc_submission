from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =========================
# 1. PATHS
# =========================
PRED_PATH = Path("models/iq_cnn1d_test_predictions.npz")
META_PATH = Path("data/processed/images/metadata.npz")
OUT_DIR = Path("models/evaluation_1d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. LOAD DATA
# =========================
pred_data = np.load(PRED_PATH, allow_pickle=True)

y_true = pred_data["y_true"]
y_pred = pred_data["y_pred"]
test_idx = pred_data["test_idx"]

metadata = np.load(META_PATH, allow_pickle=True)
snrs_all = metadata["snrs"]
class_names = metadata["class_names"].astype(str).tolist()

snrs_test = snrs_all[test_idx]

print("Number of test samples:", len(y_true))
print("Number of classes:", len(class_names))
print("Classes:", class_names)

# =========================
# 3. OVERALL METRICS
# =========================
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

print("\n===== 1D CNN OVERALL RESULTS =====")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro-F1:      {macro_f1:.4f}")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print("\n===== CLASSIFICATION REPORT =====")
print(report)

with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# =========================
# 4. CONFUSION MATRIX
# =========================
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
)

fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)
disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False,
)
ax.set_title("I/Q 1D CNN - Confusion Matrix")
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

# =========================
# 5. ACCURACY BY SNR
# =========================
unique_snrs = np.sort(np.unique(snrs_test))
accuracy_by_snr = []
macro_f1_by_snr = []

print("\n===== RESULTS BY SNR =====")
print("SNR (dB) | Accuracy | Macro-F1")

for snr in unique_snrs:
    mask = snrs_test == snr

    acc = accuracy_score(y_true[mask], y_pred[mask])
    f1 = f1_score(
        y_true[mask],
        y_pred[mask],
        average="macro",
        zero_division=0,
    )

    accuracy_by_snr.append(acc)
    macro_f1_by_snr.append(f1)

    print(f"{snr:>7} | {acc:.4f} | {f1:.4f}")

# Accuracy vs SNR plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(unique_snrs, np.array(accuracy_by_snr) * 100, marker="o")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("I/Q 1D CNN - Accuracy vs SNR")
ax.grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "accuracy_vs_snr.png", dpi=200)
plt.close(fig)

# Macro-F1 vs SNR plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(unique_snrs, macro_f1_by_snr, marker="o")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Macro-F1")
ax.set_title("I/Q 1D CNN - Macro-F1 vs SNR")
ax.grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "macro_f1_vs_snr.png", dpi=200)
plt.close(fig)

# =========================
# 6. SAVE NUMERIC RESULTS
# =========================
results = {
    "test_accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "snr_results": [
        {
            "snr_db": float(snr),
            "accuracy": float(acc),
            "macro_f1": float(f1),
        }
        for snr, acc, f1 in zip(
            unique_snrs, accuracy_by_snr, macro_f1_by_snr
        )
    ],
}

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nEvaluation completed.")
print("Results saved in:", OUT_DIR)


