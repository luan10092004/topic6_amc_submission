from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


# =========================================================
# LOCKED FINAL RESULTS
# =========================================================

rows = [
    {
        "Model": "IQ CNN 1D",
        "Accuracy": 0.7584,
        "Macro-F1": 0.7548,
        "Inference Median (ms/sample)": 0.4117,
        "Inference P95 (ms/sample)": 0.81761,
        "E2E Median (ms/sample)": 0.5150,
        "E2E P95 (ms/sample)": 0.8155,
        "Parameters": 36968,
        "Model Size (MB)": 0.15032,
        "Model Tensor Memory (MB)": 0.14275,
    },

    {
        "Model": "Spectrogram CNN 2D",
        "Accuracy": 0.5905,
        "Macro-F1": 0.5669,
        "Inference Median (ms/sample)": 0.60845,
        "Inference P95 (ms/sample)": 1.05842,
        "E2E Median (ms/sample)": 1.3390,
        "E2E P95 (ms/sample)": 3.9097,
        "Parameters": 24040,
        "Model Size (MB)": 0.10040,
        "Model Tensor Memory (MB)": 0.09258,
    },

    {
        "Model": "Constellation Density CNN 2D",
        "Accuracy": 0.6104,
        "Macro-F1": 0.6157,
        "Inference Median (ms/sample)": 0.61945,
        "Inference P95 (ms/sample)": 0.839525,
        "E2E Median (ms/sample)": 1.0131,
        "E2E P95 (ms/sample)": 2.2546,
        "Parameters": 24040,
        "Model Size (MB)": 0.10029,
        "Model Tensor Memory (MB)": 0.09258,
    },
]


output = MODEL_DIR / "final_results_complete.csv"


with open(
    output,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)


print("FINAL RESULTS LOCKED")
print("Saved:", output)

print()

for row in rows:

    print(
        f"{row['Model']:32s}"
        f" | Acc={row['Accuracy'] * 100:.2f}%"
        f" | F1={row['Macro-F1'] * 100:.2f}%"
        f" | Infer={row['Inference Median (ms/sample)']:.4f} ms"
        f" | E2E={row['E2E Median (ms/sample)']:.4f} ms"
    )
