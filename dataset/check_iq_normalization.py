from pathlib import Path
import numpy as np

DATA_PATH = Path("data/processed/radio_ml_split.npz")

data = np.load(DATA_PATH, allow_pickle=False)

X = data["X"].astype(np.float64)
train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]

print("===== DATASET =====")
print("X shape:", X.shape)
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))

# RMS chung cá»§a I vÃ  Q cho tá»«ng burst
rms = np.sqrt(np.mean(X[:, 0, :]**2 + X[:, 1, :]**2, axis=1))

# Peak amplitude cá»§a tá»«ng burst
peak = np.sqrt(X[:, 0, :]**2 + X[:, 1, :]**2).max(axis=1)

print("\n===== ORIGINAL IQ STATISTICS =====")
print("RMS min:", rms.min())
print("RMS mean:", rms.mean())
print("RMS max:", rms.max())

print("\nPeak amplitude min:", peak.min())
print("Peak amplitude mean:", peak.mean())
print("Peak amplitude max:", peak.max())

print("\n===== RMS AFTER PER-BURST NORMALIZATION =====")
print("Mean RMS after normalization:",
      np.mean(rms / np.maximum(rms, 1e-12)))
print("RMS standard deviation before normalization:",
      rms.std())

# Kiá»ƒm tra cÃ¡c táº­p cÃ³ giao nhau hay khÃ´ng
print("\n===== SPLIT OVERLAP CHECK =====")
print("Train âˆ© Validation:",
      len(np.intersect1d(train_idx, val_idx)))
print("Train âˆ© Test:",
      len(np.intersect1d(train_idx, test_idx)))
print("Validation âˆ© Test:",
      len(np.intersect1d(val_idx, test_idx)))
