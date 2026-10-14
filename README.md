# EE136IU Capstone Mini-Project — Topic 6

## Multi-Domain 2D Representation for Automatic Modulation Classification

This project investigates whether transforming short complex I/Q radio bursts from the original 1D time domain into 2D representations improves Automatic Modulation Classification (AMC).

Three pipelines are compared:

1. Raw I/Q → 1D CNN baseline
2. Raw I/Q → STFT Spectrogram → 2D CNN
3. Raw I/Q → Constellation Density → 2D CNN

The project evaluates classification performance, SNR-dependent behavior, confusion matrices, inference latency, end-to-end latency, and model footprint.

---

## 1. Main Result

For this experiment, converting the raw 1D I/Q burst into a 2D representation did **not** improve overall classification accuracy.

| Model | Accuracy | Macro-F1 | Inference Median | E2E Median | Parameters |
|---|---:|---:|---:|---:|---:|
| IQ CNN 1D | **75.84%** | **75.48%** | **0.4117 ms** | **0.5150 ms** | 36,968 |
| Spectrogram CNN 2D | 59.05% | 56.69% | 0.6085 ms | 1.3390 ms | 24,040 |
| Constellation Density CNN 2D | 61.04% | 61.57% | 0.6195 ms | 1.0131 ms | 24,040 |

The 1D CNN provides the best overall accuracy and latency.

The two 2D CNNs use fewer parameters and less model memory, but their full pipelines require additional preprocessing.

Among the two 2D representations, Constellation Density performs better overall than the Spectrogram representation.

---

## 2. Dataset

The experiments use RadioML 2016.10a style I/Q bursts.

Each burst contains:

```text
2 × 128
```

samples, where:

```text
row 0 = In-phase component (I)
row 1 = Quadrature component (Q)
```

Eight modulation classes are used:

```text
BPSK
QPSK
8PSK
QAM16
QAM64
CPFSK
GFSK
PAM4
```

The experiments cover 15 SNR levels:

```text
-10, -8, -6, -4, -2,
  0,  2,  4,  6,  8,
 10, 12, 14, 16, 18 dB
```

A total of 120,000 bursts are used in the full experiment.

The data are split into:

```text
70% training
15% validation
15% test
```

The resulting split sizes are:

```text
Train: 84,000
Validation: 18,000
Test: 18,000
```

The splits are stratified by modulation class and SNR and are index-disjoint.

---

## 3. RMS Normalization

Each I/Q burst is normalized independently using its RMS energy.

For a complex signal

```text
r[n] = I[n] + jQ[n]
```

the RMS value is

```text
E_rms = sqrt(mean(I[n]^2 + Q[n]^2))
```

and the normalized signal is

```text
r_norm[n] = r[n] / E_rms
```

This removes arbitrary burst-level amplitude scaling while retaining the signal structure used for classification.

---

## 4. 2D Representations

### 4.1 STFT Spectrogram

The complex signal is constructed as:

```text
x[n] = I[n] + jQ[n]
```

The Short-Time Fourier Transform uses:

```text
nperseg = 32
noverlap = 16
nfft = 32
```

The STFT magnitude is converted to power:

```text
P = |Z|^2
```

and then to decibels:

```text
10 log10(P + 1e-12)
```

Each spectrogram is normalized to `[0, 1]` and resized to:

```text
32 × 32
```

---

### 4.2 Constellation Density

The I/Q coordinates are first normalized using the maximum burst amplitude.

A 2D histogram is then created over:

```text
I ∈ [-1, 1]
Q ∈ [-1, 1]
```

with:

```text
32 × 32 bins
```

The histogram uses:

```text
log(1 + histogram)
```

followed by normalization to `[0, 1]`.

The resulting image represents the spatial density of the constellation points.

---

## 5. Neural Network Architectures

### 5.1 IQ CNN 1D Baseline

```text
Input [2 × 128]

Conv1D 2 → 32, kernel 7
BatchNorm
ReLU
MaxPool

Conv1D 32 → 64, kernel 5
BatchNorm
ReLU
MaxPool

Conv1D 64 → 128, kernel 3
BatchNorm
ReLU

Adaptive Average Pool
Dropout 0.3
Linear 128 → 8
```

Total parameters:

```text
36,968
```

---

### 5.2 Spectrogram / Density CNN 2D

Both 2D representations use the same lightweight CNN architecture.

```text
Input [1 × 32 × 32]

Conv2D 1 → 16, kernel 3
BatchNorm
ReLU
MaxPool

Conv2D 16 → 32, kernel 3
BatchNorm
ReLU
MaxPool

Conv2D 32 → 64, kernel 3
BatchNorm
ReLU

Adaptive Average Pool
Dropout 0.3
Linear 64 → 8
```

Total parameters:

```text
24,040
```

---

## 6. Repository Structure

```text
topic6_amc_submission/
│
├── data/
│   ├── demo/
│   │   └── demo_samples.npz
│   └── processed/
│       └── full generated data (not committed)
│
├── dataset/
│   ├── prepare_data.py
│   ├── generate_dataset.py
│   ├── create_images.py
│   ├── check_iq_normalization.py
│   └── create_demo_bundle.py
│
├── training/
│   ├── train_1d_cnn.py
│   ├── train_2d_cnn.py
│   └── train_density_cnn.py
│
├── evaluation/
│   ├── evaluate_1d_cnn.py
│   ├── evaluate_2d_cnn.py
│   ├── evaluate_density.py
│   ├── benchmark_end_to_end.py
│   ├── benchmark_inference_stable.py
│   ├── benchmark_memory.py
│   ├── make_final_figures.py
│   ├── make_macro_f1_vs_snr.py
│   ├── make_normalized_confusions.py
│   └── make_training_curves.py
│
├── demo/
│   └── app.py
│
├── models/
│   ├── iq_cnn1d.pt
│   ├── spectrogram_cnn2d.pt
│   ├── density_cnn2d.pt
│   └── final_results_complete.csv
│
├── results/
│   └── final_figures/
│
├── requirements.txt
├── run_demo.py
├── .gitignore
└── README.md
```

All commands below should be executed from the repository root.

---

## 7. Installation — Windows

Python 3 is required.

Create a virtual environment:

```powershell
python -m venv .venv
```

Install the required packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The tested environment uses:

```text
numpy==2.4.6
scipy==1.17.1
pandas==3.0.6
matplotlib==3.11.2
scikit-learn==1.9.1
torch==2.14.1
streamlit==1.65.0
psutil==7.2.2
```

---

## 8. Quick Demo

The repository contains a lightweight demonstration bundle:

```text
data/demo/demo_samples.npz
```

It contains:

```text
240 unseen test bursts
8 modulation classes
15 SNR levels
2 bursts per class per SNR
```

This allows the interactive demonstration to run without downloading the full RadioML dataset.

After installing the dependencies, launch the demo with one command:

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

The launcher automatically starts Streamlit in lightweight GitHub demo mode.

The interactive application shows:

- selected raw I/Q burst
- SNR and true modulation
- raw I/Q waveform
- raw constellation
- on-the-fly 32×32 Spectrogram
- on-the-fly 32×32 Constellation Density
- predictions from all three CNNs
- confidence scores
- class probability distributions
- preprocessing consistency verification
- final benchmark summary

The SNR filter can be used to select real test bursts from different SNR conditions.

No synthetic noise is added by the demo selector.

---

## 9. Full Dataset Mode

The application automatically switches to the full test set when the following files are available:

```text
data/processed/radio_ml_split.npz
data/processed/images/spectrograms.npy
data/processed/images/constellation_density.npy
```

When these files are not available, it automatically falls back to:

```text
data/demo/demo_samples.npz
```

The quick demo bundle is intended for lightweight reproducibility.

The full processed dataset is intentionally not included in the Git repository because of its size.

---

## 10. Preparing the Full Dataset

Place the raw dataset file at:

```text
data/raw/RML2016.10a_dict_optimized.pkl
```

Then run from the repository root:

```powershell
.\.venv\Scripts\python.exe .\dataset\prepare_data.py
```

Generate the 2D representations with the dataset-generation scripts in:

```text
dataset/
```

The resulting full processed data are stored under:

```text
data/processed/
```

Large generated arrays are excluded from Git through `.gitignore`.

---

## 11. Training

Train the 1D baseline:

```powershell
.\.venv\Scripts\python.exe .\training\train_1d_cnn.py
```

Train the Spectrogram 2D CNN:

```powershell
.\.venv\Scripts\python.exe .\training\train_2d_cnn.py
```

Train the Constellation Density 2D CNN:

```powershell
.\.venv\Scripts\python.exe .\training\train_density_cnn.py
```

The trained checkpoints are stored under:

```text
models/
```

The final checkpoints included with this repository are:

```text
models/iq_cnn1d.pt
models/spectrogram_cnn2d.pt
models/density_cnn2d.pt
```

---

## 12. Evaluation

The project evaluates more than aggregate accuracy.

The final evaluation includes:

```text
Overall Accuracy
Overall Macro-F1
Accuracy versus SNR
Macro-F1 versus SNR
Row-normalized confusion matrices
Training / validation loss curves
Training / validation accuracy curves
Model-only inference latency
End-to-end latency
Parameter count
Model tensor memory
```

The required row-normalized confusion matrices are generated for:

```text
-4 dB
+4 dB
+14 dB
```

Final figures are available under:

```text
results/final_figures/
```

---

## 13. Accuracy versus SNR

The 1D CNN achieves the best accuracy across all 15 tested SNR values.

At very low SNR, all models are strongly affected by noise.

At higher SNR, Constellation Density improves substantially and reaches approximately 87% accuracy, but the IQ CNN remains the strongest overall model.

The Spectrogram CNN shows weaker performance than the other two approaches.

---

## 14. Failure Analysis

The main observed failure modes are:

### Low SNR

Strong noise spreads the I/Q samples and degrades both temporal and spatial signal structure.

Constellation-density clusters become difficult to separate.

### QAM16 versus QAM64

QAM16 and QAM64 are among the most difficult classes.

Both are dense amplitude/phase constellations, and the short 128-sample bursts provide limited observations.

Noise further reduces the distinction between the two modulation families.

### Spectrogram Representation

The fixed STFT representation compresses the original 128-sample complex time-domain burst into a time-frequency image.

This representation may discard some raw temporal and phase information that remains directly available to the 1D CNN.

These observations are treated as likely physical explanations for the measured confusion patterns rather than as proof of a single causal mechanism.

---

## 15. Latency Benchmark

Two different latency measurements are reported.

### Model-only inference latency

This measures only:

```text
prepared tensor
      ↓
model forward pass
      ↓
network output
```

The final robust model-only medians are:

```text
IQ CNN 1D                   0.4117 ms/sample
Spectrogram CNN 2D          0.6085 ms/sample
Constellation Density CNN   0.6195 ms/sample
```

These results were measured on CPU with:

```text
1 PyTorch thread
batch size = 1
1000 samples per round
7 rounds
CPU affinity control
warm-up before timing
```

The median of the round medians is used as the robust final result.

---

### End-to-end latency

End-to-end latency additionally includes representation preprocessing.

Final median values:

```text
IQ CNN 1D                   0.5150 ms/sample
Spectrogram CNN 2D          1.3390 ms/sample
Constellation Density CNN   1.0131 ms/sample
```

The gap between inference-only and end-to-end latency demonstrates the computational cost of creating the 2D representations.

---

## 16. Model Footprint

| Model | Parameters | Model Tensor Memory |
|---|---:|---:|
| IQ CNN 1D | 36,968 | 0.14275 MB |
| Spectrogram CNN 2D | 24,040 | 0.09258 MB |
| Constellation Density CNN 2D | 24,040 | 0.09258 MB |

The 2D CNN architecture is smaller than the 1D baseline in parameter count and tensor memory.

However, lower parameter count does not automatically imply lower end-to-end latency because the 2D pipelines require additional preprocessing.

---

## 17. Training History

The selected checkpoints correspond to the best validation-accuracy epochs:

```text
IQ CNN 1D:
Best epoch = 14
Best validation accuracy = 76.02%

Spectrogram CNN 2D:
Best epoch = 17
Best validation accuracy = 59.58%

Constellation Density CNN 2D:
Best epoch = 18
Best validation accuracy = 62.13%
```

Training histories are stored under:

```text
models/
```

and the generated curves are available under:

```text
results/final_figures/
```

---

## 18. Reproducibility

The main experiment uses deterministic random seeds where applicable.

The primary project seed is:

```text
42
```

The same seed is used for operations such as deterministic demo-bundle selection and benchmark sample selection.

The data split is stratified and index-disjoint.

The available source dataset metadata does not expose a separate transmitter/channel-instance identifier, so an additional group-level leakage check based on transmitter or channel-instance identity cannot be performed.

The lightweight GitHub demo bundle contains only samples selected from the original test split.

---

## 19. Pipeline Verification

The Streamlit application regenerates the Spectrogram and Constellation Density representations directly from the selected raw I/Q burst.

It then compares the newly generated images with the saved representations.

For verified samples, the observed maximum absolute difference is:

```text
Spectrogram:
0.0000000000

Constellation Density:
0.0000000000
```

This confirms that the live demo preprocessing is consistent with the preprocessing used to generate the model inputs.

The saved images are used only for verification.

The 2D CNN inference itself uses the newly generated on-the-fly representations.

---

## 20. Important Distinction: Demo Results versus Final Metrics

Changing the selected burst in the Streamlit application changes:

```text
raw I/Q
SNR
true modulation
Spectrogram
Constellation Density
predictions
confidence
class probabilities
```

The Final Benchmark Summary does not change.

This is intentional.

The benchmark table contains aggregate controlled evaluation results from the complete experiment, not measurements from one selected burst.

---

## 21. Limitations

This project evaluates one lightweight 1D CNN architecture and one lightweight 2D CNN architecture.

The results therefore do not imply that every possible 2D representation or every possible 2D neural network will underperform a 1D model.

The experiment specifically shows that, under the tested preprocessing, architectures, dataset, and training configuration:

```text
Raw I/Q CNN 1D > Spectrogram CNN 2D
Raw I/Q CNN 1D > Constellation Density CNN 2D
```

in overall classification accuracy.

The 2D models nevertheless provide a smaller neural-network footprint and useful visual representations.

---

## 22. Final Conclusion

The experiment demonstrates that transforming short I/Q bursts into 2D images creates intuitive visual representations, but does not automatically improve AMC classification performance.

The main findings are:

```text
1. Raw I/Q CNN 1D achieves the best overall classification performance.

2. Constellation Density performs better than Spectrogram
   among the tested 2D representations.

3. The 2D CNNs use fewer parameters and less model memory.

4. 2D representation generation adds measurable
   end-to-end latency.

5. Representation choice creates a trade-off between
   visual interpretability, model footprint, latency,
   and classification accuracy.
```

---

## Quick Start Summary

From the repository root:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then launch the lightweight interactive demo:

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

The demo does not require the full RadioML dataset.