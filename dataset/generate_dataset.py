
from pathlib import Path

import numpy as np
from scipy.signal import stft
from scipy.ndimage import zoom

# ==========================================
# 1. CAU HINH
# ==========================================
DATA_PATH = Path("data/processed/radio_ml_split.npz")
OUTPUT_DIR = Path("data/processed/images")

IMAGE_SIZE = 32
BATCH_SIZE = 256


# ==========================================
# 2. SPECTROGRAM 32 x 32
# ==========================================
def make_spectrogram(iq):
    I = iq[0]
    Q = iq[1]
    signal = I + 1j * Q

    _, _, Z = stft(
        signal,
        fs=1.0,
        nperseg=32,
        noverlap=16,
        nfft=32,
        boundary=None,
        padded=False,
        return_onesided=False,
    )

    # Power spectrum in dB
    power = np.abs(Z) ** 2
    spec_db = 10 * np.log10(power + 1e-12)

    # Normalize each image to [0, 1]
    spec_db -= spec_db.min()
    max_value = spec_db.max()

    if max_value > 0:
        spec_db /= max_value

    # Resize to 32 x 32
    resized = zoom(
        spec_db,
        (
            IMAGE_SIZE / spec_db.shape[0],
            IMAGE_SIZE / spec_db.shape[1],
        ),
        order=1,
    )

    # Ensure exact output size
    result = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32,
    )

    h = min(resized.shape[0], IMAGE_SIZE)
    w = min(resized.shape[1], IMAGE_SIZE)
    result[:h, :w] = resized[:h, :w]

    return result


# ==========================================
# 3. CONSTELLATION DENSITY 32 x 32
# ==========================================
def make_constellation_density(iq):
    I = iq[0].astype(np.float32)
    Q = iq[1].astype(np.float32)

    # Normalize by the maximum amplitude of this burst
    amplitude = np.sqrt(I**2 + Q**2)
    scale = amplitude.max()

    if scale > 1e-12:
        I = I / scale
        Q = Q / scale

    hist, _, _ = np.histogram2d(
        I,
        Q,
        bins=IMAGE_SIZE,
        range=[[-1, 1], [-1, 1]],
    )

    # Compress density differences
    hist = np.log1p(hist)

    if hist.max() > 0:
        hist /= hist.max()

    return hist.T.astype(np.float32)


# ==========================================
# 4. TAO DATASET THEO BATCH
# ==========================================
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATA_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Dang doc dataset...")
    data = np.load(DATA_PATH, allow_pickle=False)

    X = data["X"]
    y = data["y"]
    snrs = data["snrs"]

    N = len(X)

    print("Tong so mau:", N)
    print("Shape X:", X.shape)

    # Create disk-backed arrays
    spec_path = OUTPUT_DIR / "spectrograms.npy"
    dens_path = OUTPUT_DIR / "constellation_density.npy"

    specs = np.lib.format.open_memmap(
        spec_path,
        mode="w+",
        dtype=np.float32,
        shape=(N, IMAGE_SIZE, IMAGE_SIZE),
    )

    densities = np.lib.format.open_memmap(
        dens_path,
        mode="w+",
        dtype=np.float32,
        shape=(N, IMAGE_SIZE, IMAGE_SIZE),
    )

    # Process samples in batches
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)

        for i in range(start, end):
            specs[i] = make_spectrogram(X[i])
            densities[i] = make_constellation_density(X[i])

        # Save progress to disk
        specs.flush()
        densities.flush()

        print(
            f"Da xu ly {end}/{N} mau "
            f"({100 * end / N:.1f}%)"
        )

    # Save labels and split indices for future training
    np.savez_compressed(
        OUTPUT_DIR / "metadata.npz",
        y=y,
        snrs=snrs,
        train_idx=data["train_idx"],
        val_idx=data["val_idx"],
        test_idx=data["test_idx"],
        class_names=data["class_names"],
    )

    # Verify output
    print("\n===== KIEM TRA =====")
    print("Spectrogram:", specs.shape)
    print("Constellation density:", densities.shape)
    print("Spectrogram min/max:", float(specs.min()),
          float(specs.max()))
    print("Density min/max:", float(densities.min()),
          float(densities.max()))

    del specs, densities

    print("\nHOAN TAT!")
    print("Thu muc output:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
