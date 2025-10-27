import os
import cv2
import numpy as np
import pytesseract
from app.services.ocr.ocr_alignment import center_digit_image
from app.services.ocr.ocr_config import OCR_CONFIG, OCR_DEBUG_DIR, DEBUG
from app.services.model_predict import predict_digit  # CNN

def prepare_for_ocr(cell: np.ndarray, out_size=160, stretch_x=1.4, debug=True, debug_name="cell") -> np.ndarray:
    """
    Prépare une cellule (fond noir, texte blanc) pour l’OCR.
    Améliore le contraste, renforce les traits et élargit horizontalement les chiffres.
    """
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # 🔸 Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(cell)
    img = cv2.bilateralFilter(img, 5, 50, 50)

    # 🔸 Binarisation (blanc sur noir)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # 🟩 Renforcer les traits
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 🔹 Étirement horizontal contrôlé
    if stretch_x != 1:
        h, w = binary.shape
        new_w = int(w * stretch_x)
        binary = cv2.resize(binary, (new_w, h), interpolation=cv2.INTER_CUBIC)

    # 🟨 Affûtage après dilation
    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    binary = cv2.filter2D(binary, -1, sharpen_kernel)

    # 🔸 Redimension finale + bordure blanche
    binary = cv2.resize(binary, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    final = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    return final


def read_cells_ocr(cells_grid, cell_ocr="tmp", mode="hybrid"):
    """
    Lit les chiffres d'une grille 3x9 d'images selon le mode spécifié :
      - 'tesseract' : uniquement Tesseract
      - 'cnn'       : uniquement le modèle CNN
      - 'hybrid'    : CNN avec fallback Tesseract si None

    Args:
        cells_grid: grille 3x9 (np.ndarray ou None)
        cell_ocr: dossier de debug (nom du carton)
        mode: str ('cnn' | 'tesseract' | 'hybrid')

    Returns:
        Liste 3x9 de nombres détectés (ou None si vide)
    """
    result = []

    for r, row in enumerate(cells_grid):
        row_values = []
        for c, cell in enumerate(row):
            if cell is None:
                row_values.append(None)
                continue

            # 🔹 Cas des colonnes avec 2 chiffres potentiels
            halves = [cell] if c == 0 else [
                cell[:, :cell.shape[1] // 2],
                cell[:, cell.shape[1] // 2:]
            ]

            digits = []
            for i, part in enumerate(halves):
                part = center_digit_image(part, debug_name=f"r{r}_c{c}_part{i}", context="ocr")
                kernel = np.ones((2, 2), np.uint8)
                part = cv2.dilate(part, kernel, iterations=1)
                prepped = prepare_for_ocr(part, debug=True, debug_name=f"r{r}_c{c}_part{i}")

                text = None

                # 🔸 MODE CNN
                if mode in ("cnn", "hybrid"):
                    digit = predict_digit(prepped)
                    text = str(digit) if digit is not None else None

                # 🔸 MODE TESSERACT ou fallback
                if (mode == "tesseract") or (mode == "hybrid" and text is None):
                    text_raw = pytesseract.image_to_string(prepped, config=OCR_CONFIG).strip()
                    text = text_raw if text_raw.isdigit() else None

                # 🔍 Debug
                if DEBUG:
                    os.makedirs(os.path.join(OCR_DEBUG_DIR, cell_ocr), exist_ok=True)
                    debug_text = text or "?"
                    debug_img = cv2.putText(
                        prepped.copy(), debug_text, (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2
                    )
                    cv2.imwrite(os.path.join(
                        OCR_DEBUG_DIR, f"{cell_ocr}/r{r}_c{c}_part{i}_{debug_text}.png"
                    ), debug_img)

                if text is not None:
                    digits.append(text)

            # 🧮 Reconstitution du nombre complet
            number = None
            if digits:
                try:
                    number = int("".join(digits))
                except ValueError:
                    number = None

            # 🔸 Ajustement selon colonne (logique du bingo)
            if number is not None:
                if number < 10 and c > 0:
                    number += c * 10
                if c == 8 and number == 0:
                    number = 90

            row_values.append(number)
        result.append(row_values)

    return result

def create_carton_comparison_image(carton_img, ocr_tess, ocr_cnn, debug_name="carton"):
    """
    Crée une image comparative : 
    - à gauche le carton redressé,
    - au centre les résultats Tesseract,
    - à droite ceux du CNN.
    """
    h, w = carton_img.shape[:2]

    # 🔹 Créer un fond blanc (même hauteur, 3 colonnes)
    canvas = np.ones((h, w * 3, 3), dtype=np.uint8) * 255

    # 🟦 Colonne 1 : image du carton redressé
    canvas[:, :w] = carton_img

    # 🟧 Colonne 2 : Tesseract
    tess_zone = canvas[:, w:w*2]
    cv2.putText(tess_zone, "Tesseract", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    draw_carton_text(tess_zone, ocr_tess, offset_x=w)

    # 🟩 Colonne 3 : CNN
    cnn_zone = canvas[:, w*2:w*3]
    cv2.putText(cnn_zone, "CNN", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 0), 2)
    draw_carton_text(cnn_zone, ocr_cnn, offset_x=w*2)

    # 🔹 Sauvegarde
    os.makedirs(OCR_DEBUG_DIR, exist_ok=True)
    path = os.path.join(OCR_DEBUG_DIR, f"{debug_name}_comparison.png")
    cv2.imwrite(path, canvas)
    print(f"[OCR DEBUG] 🧩 Image comparaison enregistrée : {path}")
    return path

def draw_carton_text(canvas, numbers_grid, offset_x=0, color=(0, 0, 0)):
    """
    Écrit les nombres d'un carton (3x9) sur une image.
    """
    if not numbers_grid:
        return

    h, w = canvas.shape[:2]
    rows, cols = len(numbers_grid), len(numbers_grid[0])
    cell_w, cell_h = w // (3 * cols), h // rows

    for r, row in enumerate(numbers_grid):
        for c, num in enumerate(row):
            if num is not None:
                x = int(offset_x + c * cell_w + cell_w / 3)
                y = int((r + 0.7) * cell_h)
                cv2.putText(canvas, str(num), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
