import torch
import json
import argparse
from PIL import Image, ImageOps
import torchvision.transforms as T

from trainer import CaptchaCNN, IMG_H, IMG_W, ResizePad, decode

MODEL_PATH = "captcha_model.pth"
VOCAB_PATH = "vocab.json"


def load_image(image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)  # normalize rotation from camera/phone exports

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # flatten transparency onto white instead of letting it go black on convert("L")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = background

    return img.convert("L")


def load_model(vocab_path=VOCAB_PATH, model_path=MODEL_PATH):
    with open(vocab_path) as f:
        vocab = json.load(f)

    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
    num_classes = vocab["num_classes"]
    blank_idx = vocab["blank_idx"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, idx_to_char, blank_idx, device


def predict(image_path, model, idx_to_char, blank_idx, device):
    transform = T.Compose([
        ResizePad(IMG_H, IMG_W),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    img = load_image(image_path)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)  # (1, T, num_classes)

    return decode(logits[0], idx_to_char, blank_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to captcha image")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--vocab", default=VOCAB_PATH)
    args = parser.parse_args()

    model, idx_to_char, blank_idx, device = load_model(args.vocab, args.model)
    print(predict(args.image, model, idx_to_char, blank_idx, device))
