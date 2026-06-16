import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import csv, os, json
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

IMG_H = 64
IMG_W = 192
MAX_LEN = 8
EPOCHS = 150
BATCH_SIZE = 128
LR = 2e-3
SAVE_DIR = "captchas"
LABELS_FILE = "labels.csv"
MODEL_PATH = "captcha_model.pth"
VOCAB_PATH = "vocab.json"


class CaptchaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            # collapse height to 1 but keep MAX_LEN columns across width, so each
            # output slot sees features from its own slice of the image instead
            # of one global average over the whole captcha.
            nn.AdaptiveAvgPool2d((1, MAX_LEN)),
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)  # shared across the MAX_LEN positions
        self.num_classes = num_classes

    def forward(self, x):
        feat = self.cnn(x)                    # (B, 256, 1, MAX_LEN)
        feat = feat.squeeze(2).permute(0, 2, 1)  # (B, MAX_LEN, 256)
        feat = self.dropout(feat)
        return self.classifier(feat)  # (B, MAX_LEN, num_classes)


class CaptchaDataset(Dataset):
    def __init__(self, labels_file, save_dir, char_to_idx, pad_idx):
        self.data = []
        with open(labels_file) as f:
            for row in csv.reader(f):
                if len(row) == 2 and 1 <= len(row[1].strip()) <= MAX_LEN:
                    self.data.append((row[0], row[1].strip()))
        self.save_dir = save_dir
        self.char_to_idx = char_to_idx
        self.pad_idx = pad_idx
        self.transform = T.Compose([
            T.Resize((IMG_H, IMG_W)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        img = Image.open(os.path.join(self.save_dir, filename)).convert("L")
        img = self.transform(img)
        indices = [self.char_to_idx[c] for c in label]
        indices += [self.pad_idx] * (MAX_LEN - len(indices))
        target = torch.tensor(indices, dtype=torch.long)
        return img, target


def build_vocab(labels_file):
    chars = set()
    with open(labels_file) as f:
        for row in csv.reader(f):
            if len(row) == 2:
                chars.update(row[1].strip())
    chars = sorted(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    pad_idx = len(chars)
    idx_to_char = {i: c for i, c in enumerate(chars)}
    idx_to_char[pad_idx] = ""  # padding/end-of-captcha marker, drops out on decode
    return char_to_idx, idx_to_char, pad_idx


def decode(logits, idx_to_char):
    indices = logits.argmax(-1).tolist()
    return "".join(idx_to_char[i] for i in indices)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    char_to_idx, idx_to_char, pad_idx = build_vocab(LABELS_FILE)
    num_classes = len(idx_to_char)  # real chars + pad/end marker

    with open(VOCAB_PATH, "w") as f:
        json.dump({
            "char_to_idx": char_to_idx,
            "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
            "num_classes": num_classes,
            "pad_idx": pad_idx,
            "max_len": MAX_LEN,
        }, f, indent=2)
    print(f"Vocab: {num_classes} classes (incl. pad) → saved to {VOCAB_PATH}")

    dataset = CaptchaDataset(LABELS_FILE, SAVE_DIR, char_to_idx, pad_idx)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=(device.type == "cuda"), persistent_workers=True,
    )
    print(f"Dataset: {len(dataset)} samples")

    model = CaptchaCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_char_correct = 0
        total = 0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)  # (B, MAX_LEN, num_classes)
            loss = criterion(logits.view(-1, num_classes), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(-1)  # (B, MAX_LEN)
            total_correct += (preds == targets).all(dim=1).sum().item()
            total_char_correct += (preds == targets).sum().item()
            total += images.size(0)

        scheduler.step()
        avg_loss = total_loss / len(loader)
        acc = total_correct / total
        char_acc = total_char_correct / (total * MAX_LEN)

        sample_pred = decode(logits[0], idx_to_char)
        sample_true = "".join(idx_to_char[i] for i in targets[0].tolist())

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  acc={acc:.1%}  char_acc={char_acc:.1%}  pred='{sample_pred}' true='{sample_true}'")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nDone. Best accuracy: {best_acc:.1%} — model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
