from pathlib import Path

import numpy as np
import torch
from torch import nn


# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = Path("data/processed/radio_ml_split.npz")
MODEL_PATH = Path("models/iq_cnn1d.pt")

CLASS_NAMES = [
    "BPSK",
    "QPSK",
    "8PSK",
    "QAM16",
    "QAM64",
    "CPFSK",
    "GFSK",
    "PAM4",
]


# ==========================================
# 2. MODEL: MATCH train_1d_cnn.py
# ==========================================
class CNN1D(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# 3. LOAD MODEL AND DATA
# ==========================================
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    device = torch.device("cpu")

    data = np.load(DATA_PATH, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    test_idx = data["test_idx"]

    # Check class order if metadata exists in this file.
    if "class_names" in data.files:
        saved_names = [
            str(name) for name in data["class_names"].tolist()
        ]
        print("Class names in dataset:", saved_names)

    model = CNN1D(num_classes=8).to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Test one sample from the existing test set.
    index = int(test_idx[0])

    x = np.array(X[index], dtype=np.float32, copy=True)
    x_tensor = torch.from_numpy(x).unsqueeze(0)

    with torch.no_grad():
        logits = model(x_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_id = int(probabilities.argmax().item())
    true_id = int(y[index])

    print("\n===== DEMO MODEL TEST =====")
    print("Sample index:", index)
    print("Input shape:", tuple(x_tensor.shape))
    print("True label:", CLASS_NAMES[true_id])
    print("Predicted label:", CLASS_NAMES[predicted_id])
    print(f"Confidence: {probabilities[predicted_id].item():.2%}")

    print("\nProbabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:8s}: {probabilities[i].item():.4f}")


if __name__ == "__main__":
    main()
