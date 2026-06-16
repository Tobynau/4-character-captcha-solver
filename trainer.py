import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import csv, os, json, difflib, random
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

IMG_H = 64
IMG_W = 320
MAX_LABEL_LEN = 10  # sanity bound on label length, not a fixed output slot count anymore
EPOCHS = 10
BATCH_SIZE = 2
ACCUM_STEPS = 64  # effective batch size = BATCH_SIZE * ACCUM_STEPS = 1024
LR = 1e-3
VAL_FRACTION = 0.1
BG_FILL = 255  # captchas are light-background/dark-text; pad/fill to match
SAVE_DIR = "captchas"
LABELS_FILE = "labels.csv"
MODEL_PATH = "captcha_model.pth"
VOCAB_PATH = "vocab.json"


class ResizePad:
    """Letterbox resize: scale by the binding dimension, then pad the rest.

    Captcha source images vary wildly in aspect ratio (3:1 up to 9:1 seen in
    practice), so a plain Resize((H, W)) stretches characters inconsistently.
    """
    def __init__(self, height, width, fill=BG_FILL):
        self.height = height
        self.width = width
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.height / h, self.width / w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("L", (self.width, self.height), self.fill)
        canvas.paste(img, (0, 0))
        return canvas


class CaptchaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            # only collapse height from here on, so width (time) keeps its
            # own resolution instead of being squashed into MAX_LEN slots
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 2, padding=0), nn.BatchNorm2d(256), nn.ReLU(),
            # height is now 1: (B, 256, 1, W/8) -> a feature per time step
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        feat = self.cnn(x)                       # (B, 256, 1, T)
        assert feat.shape[2] == 1, f"expected collapsed height of 1, got {feat.shape[2]}"
        feat = feat.squeeze(2).permute(0, 2, 1)   # (B, T, 256)
        feat, _ = self.rnn(feat)                  # (B, T, 256)
        feat = self.dropout(feat)
        return self.classifier(feat)              # (B, T, num_classes)


class CaptchaDataset(Dataset):
    def __init__(self, samples, save_dir, char_to_idx, transform):
        self.data = samples
        self.save_dir = save_dir
        self.char_to_idx = char_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        img = Image.open(os.path.join(self.save_dir, filename)).convert("L")
        img = self.transform(img)
        target = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        return img, target, label


def collate_fn(batch):
    images, targets, labels = zip(*batch)
    images = torch.stack(images)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)  # CTCLoss wants targets concatenated, not padded
    return images, targets, target_lengths, labels


def load_samples(labels_file):
    data = []
    with open(labels_file) as f:
        for row in csv.reader(f):
            if len(row) == 2 and 1 <= len(row[1].strip()) <= MAX_LABEL_LEN:
                data.append((row[0], row[1].strip()))
    return data


def build_vocab(labels_file):
    chars = set()
    with open(labels_file) as f:
        for row in csv.reader(f):
            if len(row) == 2:
                chars.update(row[1].strip())
    chars = sorted(chars)
    blank_idx = 0  # CTC blank token convention
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    idx_to_char[blank_idx] = ""
    return char_to_idx, idx_to_char, blank_idx


def decode(logits, idx_to_char, blank_idx):
    """Greedy CTC decode: collapse repeats, drop blanks."""
    indices = logits.argmax(-1).tolist()
    chars = []
    prev = None
    for idx in indices:
        if idx != blank_idx and idx != prev:
            chars.append(idx_to_char[idx])
        prev = idx
    return "".join(chars)


def run_epoch(model, loader, idx_to_char, blank_idx, device, criterion, optimizer=None, accum_steps=1):
    train = optimizer is not None
    model.train(train)
    total_loss = 0
    total_correct = 0
    total_char_ratio = 0.0
    total = 0
    sample_pred = sample_true = ""

    if train:
        optimizer.zero_grad()

    for step, (images, targets, target_lengths, labels) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)  # (B, T, num_classes)
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)  # (T, B, num_classes)
            input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if train:
                (loss / accum_steps).backward()
                is_last_batch = step == len(loader) - 1
                if (step + 1) % accum_steps == 0 or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()

        total_loss += loss.item()
        preds = [decode(logits[i], idx_to_char, blank_idx) for i in range(images.size(0))]
        for pred, true in zip(preds, labels):
            total_correct += int(pred == true)
            total_char_ratio += difflib.SequenceMatcher(None, pred, true).ratio()
        total += images.size(0)
        sample_pred, sample_true = preds[0], labels[0]

    return {
        "loss": total_loss / len(loader),
        "acc": total_correct / total,
        "char_acc": total_char_ratio / total,
        "sample_pred": sample_pred,
        "sample_true": sample_true,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    char_to_idx, idx_to_char, blank_idx = build_vocab(LABELS_FILE)
    num_classes = len(idx_to_char)  # real chars + blank

    with open(VOCAB_PATH, "w") as f:
        json.dump({
            "char_to_idx": char_to_idx,
            "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
            "num_classes": num_classes,
            "blank_idx": blank_idx,
            "max_label_len": MAX_LABEL_LEN,
        }, f, indent=2)
    print(f"Vocab: {num_classes} classes (incl. blank) → saved to {VOCAB_PATH}")

    samples = load_samples(LABELS_FILE)
    random.shuffle(samples)
    split = max(1, int(len(samples) * (1 - VAL_FRACTION)))
    train_samples, val_samples = samples[:split], samples[split:]

    train_transform = T.Compose([
        ResizePad(IMG_H, IMG_W),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5, fill=BG_FILL),
        T.GaussianBlur(3, sigma=(0.1, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    val_transform = T.Compose([
        ResizePad(IMG_H, IMG_W),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = CaptchaDataset(train_samples, SAVE_DIR, char_to_idx, train_transform)
    val_dataset = CaptchaDataset(val_samples, SAVE_DIR, char_to_idx, val_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=(device.type == "cuda"), persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        pin_memory=(device.type == "cuda"), persistent_workers=True,
        collate_fn=collate_fn,
    )
    print(f"Dataset: {len(train_dataset)} train / {len(val_dataset)} val samples")

    model = CaptchaCNN(num_classes).to(device)
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_stats = run_epoch(model, train_loader, idx_to_char, blank_idx, device, criterion, optimizer, accum_steps=ACCUM_STEPS)
        val_stats = run_epoch(model, val_loader, idx_to_char, blank_idx, device, criterion, optimizer=None)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.1%} train_char_acc={train_stats['char_acc']:.1%}  "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.1%} val_char_acc={val_stats['char_acc']:.1%}  "
            f"pred='{val_stats['sample_pred']}' true='{val_stats['sample_true']}'"
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nDone. Best val accuracy: {best_val_acc:.1%} — model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
