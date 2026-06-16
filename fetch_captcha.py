import requests
import os

URL = "https://besthouseswap.com/captcha.php"
SAVE_DIR = "captchas"
COUNT = 100

os.makedirs(SAVE_DIR, exist_ok=True)

existing = [
    f for f in os.listdir(SAVE_DIR)
    if f.endswith(".png") and f[:-4].isdigit()
]
next_num = max((int(f[:-4]) for f in existing), default=0) + 1

for i in range(COUNT):
    out_path = os.path.join(SAVE_DIR, f"{next_num + i}.png")
    response = requests.get(URL, timeout=10)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)
    print(f"[{i + 1}/{COUNT}] Saved {out_path}")
