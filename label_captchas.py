import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv
import os

SAVE_DIR = "captchas"
LABELS_FILE = "labels.csv"

def load_existing_labels():
    labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, newline="") as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    labels[row[0]] = row[1]
    return labels

def save_label(filename, text):
    labels[filename] = text
    with open(LABELS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        for fname, label in labels.items():
            writer.writerow([fname, label])

def load_images():
    files = sorted(
        [f for f in os.listdir(SAVE_DIR) if f.endswith(".png") and f[:-4].isdigit()],
        key=lambda f: int(f[:-4])
    )
    return files

def show_image(index):
    if not images:
        status_var.set("No images found in captchas/")
        return

    filename = images[index]
    path = os.path.join(SAVE_DIR, filename)
    img = Image.open(path)
    # Scale up for visibility (captchas are usually small)
    scale = max(1, 300 // max(img.width, img.height))
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo

    existing = labels.get(filename, "")
    entry_var.set(existing)
    entry.focus()
    entry.icursor(tk.END)

    labeled_count = sum(1 for f in images if f in labels)
    status_var.set(f"{filename}   [{index + 1}/{len(images)}]   labeled: {labeled_count}/{len(images)}")

def submit(event=None):
    global current_index
    filename = images[current_index]
    text = entry_var.get().strip()
    if text:
        save_label(filename, text)
    if current_index < len(images) - 1:
        current_index += 1
        show_image(current_index)

def go_prev(event=None):
    global current_index
    if current_index > 0:
        current_index -= 1
        show_image(current_index)

def go_next(event=None):
    global current_index
    if current_index < len(images) - 1:
        current_index += 1
        show_image(current_index)

labels = load_existing_labels()
images = load_images()
current_index = 0

# Start on first unlabeled image
for i, f in enumerate(images):
    if f not in labels:
        current_index = i
        break

root = tk.Tk()
root.title("Captcha Labeler")
root.resizable(False, False)

image_label = tk.Label(root, bg="white", width=300, height=150)
image_label.pack(padx=20, pady=20)

entry_var = tk.StringVar()
entry = ttk.Entry(root, textvariable=entry_var, font=("Courier", 16), width=20, justify="center")
entry.pack(pady=(0, 10))
entry.bind("<Return>", submit)
entry.bind("<Left>", go_prev)
entry.bind("<Right>", go_next)

status_var = tk.StringVar()
tk.Label(root, textvariable=status_var, fg="gray").pack(pady=(0, 10))

btn_frame = tk.Frame(root)
btn_frame.pack(pady=(0, 15))
ttk.Button(btn_frame, text="← Prev", command=go_prev).grid(row=0, column=0, padx=5)
ttk.Button(btn_frame, text="Save & Next →", command=submit).grid(row=0, column=1, padx=5)

show_image(current_index)
root.mainloop()
