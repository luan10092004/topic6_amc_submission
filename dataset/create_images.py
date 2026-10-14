
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.ndimage import zoom

# =====================================
# 1. Cau hinh
# =====================================
DATA_PATH = Path("data/processed/radio_ml_split.npz")
OUTPUT_DIR = Path("data/processed/preview")

IMAGE_SIZE = 32
NUM_PREVIEW = 8

# =====================================
# 2. Tao spectrogram 32 x 32
# =====================================
def make_spectrogram(iq):
    """
    iq shape: (2, 128)
    Output shape: (32, 32)
    """
    I = iq[0]
    Q = iq[1]
    signal = I + 1j * Q

    train_2d_cnn.py_, _, Z = stft(
    signal,
    fs=1.0,
    nperseg=32,
    noverlap=16,
    nfft=32,
    boundary=None,
    padded=False,
    return_onesided=False,
)

    power = np.abs(Z) ** 2
    spec_db = 10 * np.log10(power + 1e-12)

    # Chuan hoa rieng tung anh
    spec_db = spec_db - spec_db.min()
    max_value = spec_db.max()

    if max_value > 0:
        spec_db = spec_db / max_value

    # STFT o tren co the chua it hon 32 cot.
    # Resize ve 32 x 32.
    resized = zoom(
        spec_db,
        (
            IMAGE_SIZE / spec_db.shape[0],
            IMAGE_SIZE / spec_db.shape[1],
        ),
        order=1,
    )

    resized = resized[:IMAGE_SIZE, :IMAGE_SIZE]

    # Bao dam dung kich thuoc
    result = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    h, w = resized.shape
    result[:h, :w] = resized

    return result


# =====================================
# 3. Tao constellation density 32 x 32
# =====================================

def make_constellation_density(iq):
    """
    iq shape: (2, 128)
    Output shape: (32, 32)
    """
    I = iq[0].astype(np.float32)
    Q = iq[1].astype(np.float32)

    # Chuan hoa theo bien do lon nhat cua burst
    amplitude = np.sqrt(I**2 + Q**2)
    scale = np.max(amplitude)

    if scale > 1e-12:
        I = I / scale
        Q = Q / scale

    # Histogram 2D tren mien I/Q da chuan hoa
    hist, _, _ = np.histogram2d(
        I,
        Q,
        bins=32,
        range=[[-1, 1], [-1, 1]],
    )

    # Nen su khac biet mat do
    hist = np.log1p(hist)

    if hist.max() > 0:
        hist = hist / hist.max()

    return hist.T.astype(np.float32)


# =====================================
# 4. Chay thu tren mot so mau
# =====================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Dang doc dataset da chia...")

    data = np.load(DATA_PATH, allow_pickle=False)

    X = data["X"]
    y = data["y"]
    snrs = data["snrs"]
    class_names = data["class_names"]

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(X),
        size=min(NUM_PREVIEW, len(X)),
        replace=False,
    )

    for index in sample_indices:
        iq = X[index]

        spec = make_spectrogram(iq)
        density = make_constellation_density(iq)

        assert spec.shape == (32, 32)
        assert density.shape == (32, 32)

        label = str(class_names[int(y[index])])
        snr = int(snrs[index])

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(
            spec,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[0].set_title("Spectrogram 32x32")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Frequency")

        axes[1].imshow(
            density,
            origin="lower",
            aspect="equal",
            extent=(-1, 1, -1, 1),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Constellation Density 32x32")
        axes[1].set_xlabel("I")
        axes[1].set_ylabel("Q")

        fig.suptitle(f"Class: {label} | SNR: {snr} dB")
        fig.tight_layout()

        output_file = OUTPUT_DIR / f"sample_{index}.png"
        fig.savefig(output_file, dpi=150)
        plt.close(fig)

        print(
            f"Sample {index}: class={label}, SNR={snr} dB, "
            f"Spectrogram={spec.shape}, Density={density.shape}"
        )

    print("\nHoan tat tao anh xem truoc!")
    print("Thu muc:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
