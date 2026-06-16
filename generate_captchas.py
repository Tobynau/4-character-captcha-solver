import argparse
import csv
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SAVE_DIR = "captchas"
LABELS_FILE = "labels.csv"

CHARSET = "23456789abcdefghkmnpqrstuvwxyz"  # excludes ambiguous 0/1/i/j/l/o
MIN_LEN, MAX_LEN = 4, 8

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-BoldOblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Italic.otf",
]
FONT_PATHS = [p for p in FONT_PATHS if os.path.exists(p)]

IMG_H = 80
CHAR_CELL = 32  # nominal horizontal space per character before padding

COMPLEXITY_LEVELS = ["simple", "medium", "hard"]


def random_color(low=0, high=255):
    return tuple(random.randint(low, high) for _ in range(3))


def background(width, height, complexity):
    base = random_color(200, 255)
    img = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(img)

    if complexity == "simple":
        return img, draw

    # subtle gradient/noise tint
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-12, 12, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    if complexity == "hard":
        for _ in range(random.randint(2, 4)):
            color = random_color(150, 230)
            draw.rectangle(
                [random.randint(0, width // 2), random.randint(0, height // 2),
                 random.randint(width // 2, width), random.randint(height // 2, height)],
                outline=None, fill=color,
            )
        arr = np.array(img).astype(np.int16)
        noise = np.random.randint(-15, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

    return img, draw


def draw_noise_lines(draw, width, height, complexity):
    if complexity == "simple":
        n_lines = random.randint(0, 1)
    elif complexity == "medium":
        n_lines = random.randint(2, 4)
    else:
        n_lines = random.randint(4, 7)

    for _ in range(n_lines):
        color = random_color(80, 180)
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        if random.random() < 0.5:
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 2))
        else:
            ctrl = (random.randint(0, width), random.randint(0, height))
            draw.line([(x1, y1), ctrl, (x2, y2)], fill=color, width=random.randint(1, 2))


def draw_noise_dots(draw, width, height, complexity):
    if complexity == "simple":
        n_dots = random.randint(10, 30)
    elif complexity == "medium":
        n_dots = random.randint(40, 90)
    else:
        n_dots = random.randint(100, 200)

    for _ in range(n_dots):
        x, y = random.randint(0, width), random.randint(0, height)
        r = random.randint(0, 1)
        color = random_color(60, 200)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def render_char(ch, font_path, font_size, color):
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(ch)
    w = bbox[2] - bbox[0] + 8
    h = bbox[3] - bbox[1] + 8
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((-bbox[0] + 4, -bbox[1] + 4), ch, font=font, fill=color + (255,))
    return tile


def wave_distort(img, amplitude, period):
    arr = np.array(img)
    h, w = arr.shape[:2]
    out = np.zeros_like(arr)
    for y in range(h):
        shift = int(amplitude * math.sin(2 * math.pi * y / period))
        out[y] = np.roll(arr[y], shift, axis=0)
    return Image.fromarray(out)


def generate_one(label, complexity):
    length = len(label)
    width = max(120, length * CHAR_CELL + 40)

    img, draw = background(width, IMG_H, complexity)

    if complexity != "simple":
        draw_noise_lines(draw, width, IMG_H, complexity)
    draw_noise_dots(draw, width, IMG_H, complexity)

    x = random.randint(10, 20)
    for ch in label:
        font_path = random.choice(FONT_PATHS)
        font_size = random.randint(34, 46) if complexity != "simple" else random.randint(36, 42)
        color = random_color(0, 90)
        tile = render_char(ch, font_path, font_size, color)

        if complexity != "simple":
            angle = random.uniform(-30, 30)
        else:
            angle = random.uniform(-8, 8)
        tile = tile.rotate(angle, expand=True, resample=Image.BICUBIC)

        y = random.randint(4, max(4, IMG_H - tile.height - 4))
        jitter = random.randint(-4, 4) if complexity != "simple" else 0
        img.paste(tile, (x + jitter, y), tile)

        x += tile.width - random.randint(6, 14)

    img = img.crop((0, 0, max(width, x + 20), IMG_H))

    if complexity != "simple":
        draw2 = ImageDraw.Draw(img)
        draw_noise_lines(draw2, img.width, IMG_H, complexity)

    if complexity == "hard":
        img = wave_distort(img, amplitude=random.randint(2, 4), period=random.randint(20, 40))

    if complexity == "hard" and random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.9)))
    elif complexity == "medium" and random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    return img.convert("L")


def next_index(save_dir):
    existing = [f for f in os.listdir(save_dir) if f.endswith(".png") and f[:-4].isdigit()]
    return max((int(f[:-4]) for f in existing), default=0) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--labels-file", default=LABELS_FILE)
    parser.add_argument("--min-len", type=int, default=MIN_LEN)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    start = next_index(args.save_dir)

    rows = []
    for i in range(args.count):
        length = random.randint(args.min_len, args.max_len)
        label = "".join(random.choice(CHARSET) for _ in range(length))
        complexity = random.choice(COMPLEXITY_LEVELS)

        img = generate_one(label, complexity)
        filename = f"{start + i}.png"
        img.save(os.path.join(args.save_dir, filename))
        rows.append((filename, label))

        if (i + 1) % 200 == 0:
            print(f"[{i + 1}/{args.count}] generated")

    with open(args.labels_file, "a", newline="") as f:
        writer = csv.writer(f)
        for filename, label in rows:
            writer.writerow([filename, label])

    print(f"Done. Wrote {len(rows)} images to {args.save_dir}/ and appended labels to {args.labels_file}")


if __name__ == "__main__":
    main()
