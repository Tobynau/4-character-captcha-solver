import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import csv, os, json

IMG_H = 64
IMG_W = 128
CAPTCHA_LEN = 4
EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-3
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
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, CAPTCHA_LEN * num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        feat = self.cnn(x).flatten(1)
        feat = self.dropout(feat)
        out = self.fc(feat)
        return out.view(-1, CAPTCHA_LEN, self.num_classes)  # (B, 4, num_classes)


class CaptchaDataset(Dataset):
    def __init__(self, labels_file, save_dir, char_to_idx):
        self.data = []
        with open(labels_file) as f:
            for row in csv.reader(f):
                if len(row) == 2 and len(row[1].strip()) == CAPTCHA_LEN:
                    self.data.append((row[0], row[1].strip()))
        self.save_dir = save_dir
        self.char_to_idx = char_to_idx
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
        target = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        return img, target


def build_vocab(labels_file):
    chars = set()
    with open(labels_file) as f:
        for row in csv.reader(f):
            if len(row) == 2:
                chars.update(row[1].strip())
    chars = sorted(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_idx, idx_to_char


def decode(logits, idx_to_char):
    indices = logits.argmax(-1).tolist()
    return "".join(idx_to_char[i] for i in indices)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    char_to_idx, idx_to_char = build_vocab(LABELS_FILE)
    num_classes = len(char_to_idx)

    with open(VOCAB_PATH, "w") as f:
        json.dump({
            "char_to_idx": char_to_idx,
            "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
            "num_classes": num_classes,
        }, f, indent=2)
    print(f"Vocab: {num_classes} chars → saved to {VOCAB_PATH}")

    dataset = CaptchaDataset(LABELS_FILE, SAVE_DIR, char_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
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
        total = 0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)  # (B, 4, num_classes)
            loss = criterion(logits.view(-1, num_classes), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(-1)  # (B, 4)
            total_correct += (preds == targets).all(dim=1).sum().item()
            total += images.size(0)

        scheduler.step()
        avg_loss = total_loss / len(loader)
        acc = total_correct / total

        sample_pred = decode(logits[0], idx_to_char)
        sample_true = "".join(idx_to_char[i] for i in targets[0].tolist())

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  acc={acc:.1%}  pred='{sample_pred}' true='{sample_true}'")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nDone. Best accuracy: {best_acc:.1%} — model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
