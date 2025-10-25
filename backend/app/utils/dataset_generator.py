import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# =========================
# ⚙️ CONFIGURATION
# =========================
FONT_PATH = "fonts/Rockwell.ttf"  # 👉 adapte le chemin si besoin
OUTPUT_DIR = "training/digit_dataset"
IMAGE_SIZE = (64, 64)
SAMPLES_PER_DIGIT = 500  # ↑ plus d’exemples = meilleure robustesse

# =========================
# 🧰 FONCTIONS D’AUGMENTATION
# =========================
def add_noise(img: Image.Image, amount=0.02):
    """Ajoute un bruit aléatoire sur l'image."""
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, amount, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def random_rotation(img: Image.Image, max_angle=10):
    """Applique une rotation aléatoire (±max_angle°)."""
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

def random_scale(img: Image.Image, scale_range=(0.9, 1.1)):
    """Agrandit ou rétrécit légèrement le chiffre."""
    scale = random.uniform(*scale_range)
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = img.resize((new_w, new_h), Image.LANCZOS)
    result = Image.new("L", (w, h), color=255)
    x = (w - new_w) // 2
    y = (h - new_h) // 2
    result.paste(scaled, (x, y))
    return result

def random_contrast_brightness(img: Image.Image):
    """Ajuste légèrement contraste et luminosité."""
    enh_contrast = ImageEnhance.Contrast(img)
    enh_brightness = ImageEnhance.Brightness(img)
    img = enh_contrast.enhance(random.uniform(0.8, 1.3))
    img = enh_brightness.enhance(random.uniform(0.8, 1.2))
    return img

# =========================
# 🎨 GÉNÉRATION D’UN CHIFFRE
# =========================
def generate_digit_image(digit, font_path=FONT_PATH, out_size=IMAGE_SIZE):
    img = Image.new("L", out_size, color=255)
    draw = ImageDraw.Draw(img)

    font_size = random.randint(36, 56)
    font = ImageFont.truetype(font_path, font_size)

    w, h = draw.textsize(str(digit), font=font)
    x = (out_size[0] - w) // 2 + random.randint(-4, 4)
    y = (out_size[1] - h) // 2 + random.randint(-4, 4)
    color = random.randint(0, 40)
    draw.text((x, y), str(digit), font=font, fill=color)

    # 🔸 Ajout d’effets aléatoires
    if random.random() < 0.6:
        img = random_rotation(img)
    if random.random() < 0.5:
        img = random_scale(img)
    if random.random() < 0.7:
        img = random_contrast_brightness(img)
    if random.random() < 0.4:
        img = add_noise(img, amount=random.uniform(0.01, 0.05))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    return img

# =========================
# 🚀 GÉNÉRATION DU DATASET
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for digit in range(10):
    digit_dir = os.path.join(OUTPUT_DIR, str(digit))
    os.makedirs(digit_dir, exist_ok=True)

    for i in range(SAMPLES_PER_DIGIT):
        img = generate_digit_image(digit)
        img.save(os.path.join(digit_dir, f"{digit}_{i}.png"))

print(f"✅ Dataset enrichi généré dans : {OUTPUT_DIR}")
