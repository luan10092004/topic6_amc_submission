
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. CAU HINH
# ==========================================
DATA_DIR = Path("data/processed/images")
MODEL_DIR = Path("models")

IMAGE_PATH = DATA_DIR / "constellation_density.npy"
META_PATH = DATA_DIR / "metadata.npz"

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 5
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
class SpectrogramDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])

        # Copy mot anh tu memmap vao Tensor
        image = np.array(
            self.images[index],
            dtype=np.float32,
            copy=True,
        )

        image = torch.from_numpy(image).unsqueeze(0)
        label = int(self.labels[index])

        return image, label


# ==========================================
# 3. MO HINH 2D CNN
# ==========================================
class CNN2D(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# 4. LOAD DU LIEU
# ==========================================
def main():
    print("Dang doc metadata...")

    metadata = np.load(META_PATH, allow_pickle=False)

    y = metadata["y"]
    train_idx = metadata["train_idx"]
    val_idx = metadata["val_idx"]
    test_idx = metadata["test_idx"]
    class_names = metadata["class_names"].tolist()

    # Doc anh tu dia, khong nap ca mang vao RAM ngay
    images = np.load(IMAGE_PATH, mmap_mode="r")

    print("Shape images:", images.shape)
    print("So lop:", len(class_names))
    print("Train:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))

    train_ds = SpectrogramDataset(images, y, train_idx)
    val_ds = SpectrogramDataset(images, y, val_idx)
    test_ds = SpectrogramDataset(images, y, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ======================================
    # 5. TRAINING
    # ======================================
    model = CNN2D(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_count = 0
    history = []

    model_path = MODEL_DIR / "density_cnn2d.pt"

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

        # Save model with best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_count = 0

            torch.save(
                model.state_dict(),
                model_path,
            )
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    with open(
        MODEL_DIR / "density_training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2)

    # ======================================
    # 6. TEST ONCE USING BEST MODEL
    # ======================================
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            predictions = logits.argmax(dim=1)

            test_correct += (
                predictions == yb
            ).sum().item()
            test_total += len(yb)

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_targets.extend(
                yb.cpu().numpy().tolist()
            )

    test_acc = test_correct / test_total

    print("\n===== KET QUA CUOI =====")
    print("Best epoch:", best_epoch)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Model saved:", model_path)

    # Save predictions for later analysis
    np.savez_compressed(
        MODEL_DIR / "density_test_predictions.npz",
        y_true=np.asarray(all_targets),
        y_pred=np.asarray(all_predictions),
        test_idx=test_idx,
        class_names=np.asarray(class_names),
    )


if __name__ == "__main__":
    main()
