from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "results" / "final_figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. LOAD FINAL SUMMARY
# =========================================================

results_path = MODEL_DIR / "final_results.csv"

results = pd.read_csv(
    results_path
)

print("Loaded:")
print(results)


# =========================================================
# 3. MODEL NAMES
# =========================================================

models = [
    "IQ CNN 1D",
    "Spectrogram CNN 2D",
    "Constellation Density CNN 2D",
]


# =========================================================
# 4. ACCURACY + MACRO-F1
# =========================================================

accuracy = (
    results["Accuracy"].to_numpy()
    * 100
)

macro_f1 = (
    results["Macro-F1"].to_numpy()
    * 100
)

x = np.arange(
    len(models)
)

width = 0.35

fig, ax = plt.subplots(
    figsize=(10, 6)
)

bars1 = ax.bar(
    x - width / 2,
    accuracy,
    width,
    label="Accuracy",
)

bars2 = ax.bar(
    x + width / 2,
    macro_f1,
    width,
    label="Macro-F1",
)

ax.set_ylabel(
    "Score (%)"
)

ax.set_title(
    "Overall Classification Performance"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    models,
    rotation=15,
    ha="right",
)

ax.set_ylim(
    0,
    100,
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3,
)

ax.bar_label(
    bars1,
    fmt="%.2f",
    padding=3,
)

ax.bar_label(
    bars2,
    fmt="%.2f",
    padding=3,
)

fig.tight_layout()

output = (
    FIGURE_DIR
    / "01_accuracy_macro_f1.png"
)

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", output)


# =========================================================
# 5. END-TO-END LATENCY
# =========================================================

latency = results[
    "Median E2E Latency (ms/sample)"
].to_numpy()

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    models,
    latency,
)

ax.set_ylabel(
    "Median latency (ms / sample)"
)

ax.set_title(
    "End-to-End Latency on CPU"
)

ax.tick_params(
    axis="x",
    rotation=15,
)

ax.grid(
    axis="y",
    alpha=0.3,
)

ax.bar_label(
    bars,
    fmt="%.4f",
    padding=3,
)

fig.tight_layout()

output = (
    FIGURE_DIR
    / "02_end_to_end_latency.png"
)

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", output)


# =========================================================
# 6. NUMBER OF PARAMETERS
# =========================================================

parameters = results[
    "Parameters"
].to_numpy()

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    models,
    parameters,
)

ax.set_ylabel(
    "Number of parameters"
)

ax.set_title(
    "Model Complexity"
)

ax.tick_params(
    axis="x",
    rotation=15,
)

ax.grid(
    axis="y",
    alpha=0.3,
)

ax.bar_label(
    bars,
    fmt="%d",
    padding=3,
)

fig.tight_layout()

output = (
    FIGURE_DIR
    / "03_model_parameters.png"
)

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", output)


# =========================================================
# 7. MODEL TENSOR MEMORY
# =========================================================

memory = results[
    "Model Tensor Memory (MB)"
].to_numpy()

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    models,
    memory,
)

ax.set_ylabel(
    "Tensor memory (MB)"
)

ax.set_title(
    "Model Tensor Memory"
)

ax.tick_params(
    axis="x",
    rotation=15,
)

ax.grid(
    axis="y",
    alpha=0.3,
)

ax.bar_label(
    bars,
    fmt="%.5f",
    padding=3,
)

fig.tight_layout()

output = (
    FIGURE_DIR
    / "04_model_memory.png"
)

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", output)


# =========================================================
# 8. ACCURACY VS SNR
# Values from final evaluation
# =========================================================

snr_levels = np.array([
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


iq_accuracy = np.array([
    0.2025,
    0.3350,
    0.5475,
    0.6892,
    0.7675,
    0.8608,
    0.8808,
    0.8767,
    0.8892,
    0.8858,
    0.8892,
    0.8925,
    0.8833,
    0.8883,
    0.8875,
])


spectrogram_accuracy = np.array([
    0.1800,
    0.2783,
    0.4258,
    0.4800,
    0.5425,
    0.6100,
    0.6517,
    0.6783,
    0.6933,
    0.7033,
    0.7300,
    0.7325,
    0.7358,
    0.7208,
    0.6950,
])


density_accuracy = np.array([
    0.1325,
    0.1492,
    0.2442,
    0.3467,
    0.3850,
    0.4983,
    0.6467,
    0.7783,
    0.8192,
    0.8425,
    0.8517,
    0.8642,
    0.8617,
    0.8658,
    0.8708,
])


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    snr_levels,
    iq_accuracy * 100,
    marker="o",
    label="IQ CNN 1D",
)

ax.plot(
    snr_levels,
    spectrogram_accuracy * 100,
    marker="s",
    label="Spectrogram CNN 2D",
)

ax.plot(
    snr_levels,
    density_accuracy * 100,
    marker="^",
    label="Constellation Density CNN 2D",
)

ax.set_xlabel(
    "SNR (dB)"
)

ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_title(
    "Classification Accuracy vs SNR"
)

ax.set_xticks(
    snr_levels
)

ax.set_ylim(
    0,
    100,
)

ax.grid(
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

output = (
    FIGURE_DIR
    / "05_accuracy_vs_snr.png"
)

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", output)


# =========================================================
# 9. ACCURACY VS LATENCY TRADE-OFF
# =========================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

for i, model in enumerate(models):

    ax.scatter(
        latency[i],
        accuracy[i],
        s=100,
    )

    ax.annotate(
        model,
        (
            latency[i],
            accuracy[i],
        ),
        xytext=(8, 5),
        textcoords="offset points",
    )

ax.set_xlabel(
    "Median End-to-End Latency (ms/sample)"
)

ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_title(
    "Accuracy vs End-to-End Latency"
)

ax.grid(
    alpha=0.3,
)

fig.tight_layout()

output = (
    FIGURE_DIR
    / "06_accuracy_latency_tradeoff.png"
)

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", output)


# =========================================================
# DONE
# =========================================================

print()
print("====================================")
print("FINAL FIGURES COMPLETED")
print("====================================")
print("Output folder:")
print(FIGURE_DIR)
