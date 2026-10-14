
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = Path("data/processed/radio_ml_split.npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Thiet bi huan luyen:", device)


# ==========================================
# 2. DATASET
# ==========================================
class IQDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])

        # Input shape: (2, 128), channels = I and Q
        x = np.array(
            self.X[index],
            dtype=np.float32,
            copy=True,
        )
        x = torch.from_numpy(x)

        label = int(self.y[index])
        return x, label


# ==========================================
# 3. 1D CNN MODEL
# ==========================================
class CNN1D(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
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
# 4. LOAD DATA
# ==========================================
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATA_PATH}"
        )

    print("Dang doc dataset I/Q...")
    data = np.load(DATA_PATH, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Train:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))

    assert X.ndim == 3 and X.shape[1:] == (2, 128), (
        "X phai co shape (N, 2, 128)"
    )

    train_ds = IQDataset(X, y, train_idx)
    val_ds = IQDataset(X, y, val_idx)
    test_ds = IQDataset(X, y, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ======================================
    # 5. TRAIN
    # ======================================
    model = CNN1D(num_classes=8).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_count = 0
    history = []

    model_path = MODEL_DIR / "iq_cnn1d.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(yb)
            train_correct += (
                logits.argmax(dim=1) == yb
            ).sum().item()
            train_total += len(yb)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss_sum += loss.item() * len(yb)
                val_correct += (
                    logits.argmax(dim=1) == yb
                ).sum().item()
                val_total += len(yb)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_count = 0

            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    with open(
        MODEL_DIR / "iq_cnn1d_training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2)

    # ======================================
    # 6. TEST BEST MODEL
    # ======================================
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)

            logits = model(xb)
            predictions = logits.argmax(dim=1)

            y_true.extend(yb.numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\n===== FINAL 1D CNN RESULTS =====")
    print("Best epoch:", best_epoch)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Model saved:", model_path)

    np.savez_compressed(
        MODEL_DIR / "iq_cnn1d_test_predictions.npz",
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        test_idx=test_idx,
    )


if __name__ == "__main__":
    main()
