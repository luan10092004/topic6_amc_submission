
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# 1. Cau hinh dataset
# =========================
DATA_PATH = Path("data/raw/RML2016.10a_dict_optimized.pkl")
OUTPUT_DIR = Path("data/processed")

SELECTED_CLASSES = [
    "BPSK",
    "QPSK",
    "8PSK",
    "QAM16",
    "QAM64",
    "CPFSK",
    "GFSK",
    "PAM4",
]

SELECTED_SNRS = list(range(-10, 20, 2))

RANDOM_STATE = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Doc dataset
    print("Dang doc dataset...")

    with open(DATA_PATH, "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    X_list = []
    y_list = []
    snr_list = []

    # 3. Loc cac lop va muc SNR
    print("Dang loc du lieu...")

    for (modulation, snr), samples in dataset.items():
        if modulation not in SELECTED_CLASSES:
            continue

        if snr not in SELECTED_SNRS:
            continue

        # samples: (1000, 2, 128)
        X_list.append(samples.astype(np.float32))
        y_list.extend([modulation] * len(samples))
        snr_list.extend([snr] * len(samples))

    if not X_list:
        raise ValueError(
            "Khong tim thay du lieu. "
            "Hay kiem tra DATA_PATH va ten cac lop dieu che."
        )

    X = np.concatenate(X_list, axis=0)
    y = np.asarray(y_list)
    snrs = np.asarray(snr_list, dtype=np.int16)

        
    # RMS normalization for each individual I/Q burst
    # X shape: (N, 2, 128)

    epsilon = 1e-12

    rms = np.sqrt(
    np.mean(
        X[:, 0, :] ** 2 + X[:, 1, :] ** 2,
        axis=1,
        keepdims=True,
        )
    )

    # rms shape: (N, 1); reshape to (N, 1, 1)
    rms = rms[:, :, np.newaxis]

    X = X / np.maximum(rms, epsilon).astype(np.float32)

    print("\n===== RMS NORMALIZATION CHECK =====")
    rms_after = np.sqrt(
    np.mean(
        X[:, 0, :] ** 2 + X[:, 1, :] ** 2,
        axis=1,
        )
    )
    print("RMS min:", float(rms_after.min()))
    print("RMS mean:", float(rms_after.mean()))
    print("RMS max:", float(rms_after.max()))


    # 4. Ma hoa nhan thanh so nguyen
    class_names = np.array(SELECTED_CLASSES)
    class_to_id = {
        name: index for index, name in enumerate(class_names)
    }

    y_encoded = np.array(
        [class_to_id[name] for name in y],
        dtype=np.int64,
    )

    # 5. Chia tap 70% train, 15% validation, 15% test

    # 5. Chia rieng tung nhom modulation + SNR
    train_parts = []
    val_parts = []
    test_parts = []

    for modulation in SELECTED_CLASSES:
        for snr in SELECTED_SNRS:
            group_idx = np.where(
                (y == modulation) & (snrs == snr)
            )[0]

            group_train, group_temp = train_test_split(
                group_idx,
                test_size=0.30,
                random_state=RANDOM_STATE,
            )

            group_val, group_test = train_test_split(
                group_temp,
                test_size=0.50,
                random_state=RANDOM_STATE,
            )

            train_parts.append(group_train)
            val_parts.append(group_val)
            test_parts.append(group_test)

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)

    # Tron thu tu index trong tung tap
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    
    print("\n===== KIEM TRA PHAN BO SNR =====")

    for name, idx in [
        ("Train", train_idx),
        ("Validation", val_idx),
        ("Test", test_idx),
    ]:
        print(f"\n{name}:")
        for snr in SELECTED_SNRS:
            count = np.sum(snrs[idx] == snr)
            print(f"  SNR {snr:>3} dB: {count}")

    # 6. Luu du lieu
    np.savez_compressed(
        OUTPUT_DIR / "radio_ml_split.npz",
        X=X,
        y=y_encoded,
        snrs=snrs,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        class_names=class_names,
    )

    print("\n===== KET QUA =====")
    print("Tong so mau:", len(X))
    print("Shape X:", X.shape)
    print("Shape y:", y_encoded.shape)
    print("So lop:", len(class_names))
    print("Cac lop:", class_names.tolist())
    print("Cac SNR:", SELECTED_SNRS)
    print("Train:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))
    print("Da luu:", OUTPUT_DIR / "radio_ml_split.npz")


if __name__ == "__main__":
    main()

    
