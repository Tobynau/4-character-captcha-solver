import torch
import json
import argparse
from PIL import Image
import torchvision.transforms as T

from trainer import CaptchaCNN, IMG_H, IMG_W, decode

MODEL_PATH = "captcha_model.pth"
VOCAB_PATH = "vocab.json"


def load_model(vocab_path=VOCAB_PATH, model_path=MODEL_PATH):
    with open(vocab_path) as f:
        vocab = json.load(f)

    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
    num_classes = vocab["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, idx_to_char, device


def predict(image_path, model, idx_to_char, device):
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)  # (1, MAX_LEN, num_classes)

    return decode(logits[0], idx_to_char)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to captcha image")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--vocab", default=VOCAB_PATH)
    args = parser.parse_args()

    model, idx_to_char, device = load_model(args.vocab, args.model)
    print(predict(args.image, model, idx_to_char, device))
