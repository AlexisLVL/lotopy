import os
import cv2
import numpy as np
import pytesseract
from fastapi import UploadFile
from skimage.measure import shannon_entropy

# ============================================================
# 🧩 CONFIGURATION GLOBALE
# ============================================================

OCR_CONFIG = (
    "--oem 1 "
    "--psm 13 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c load_system_dawg=false "
    "-c load_freq_dawg=false "
    "-c classify_bln_numeric_mode=1 "
    "-c tessedit_adaption_debug=0 "
)

DEBUG = True
if DEBUG:
    BASE_DIR = "detected_cartons"
    OCR_DEBUG_DIR = os.path.join(BASE_DIR, "ocr_debug")

    for d in [BASE_DIR, OCR_DEBUG_DIR]:
        os.makedirs(d, exist_ok=True)


# ============================================================
# ⚙️  OUTILS GÉNÉRAUX
# ============================================================

def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Lit une image envoyée depuis FastAPI."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def save_image(img: np.ndarray, path: str) -> None:
    """Sauvegarde une image au format PNG."""
    cv2.imwrite(path, img)

# ============================================================
# 🧠  PRÉTRAITEMENT
# ============================================================

def preprocess_gray(img: np.ndarray) -> np.ndarray:
    """Convertit en niveaux de gris et améliore le contraste."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def threshold_binary_inv(gray: np.ndarray) -> np.ndarray:
    """Applique une binarisation inversée (fond noir, texte blanc)."""
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    return binary


def extract_grid_mask(binary: np.ndarray) -> np.ndarray:
    """Détecte les lignes horizontales et verticales du carton."""
    h, w = binary.shape
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    hor_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    ver_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    return cv2.bitwise_or(hor_lines, ver_lines)


def filter_contours(contours, w_img, h_img):
    """Filtre les contours selon la taille et le ratio typique d’un carton."""
    filtered = []
    min_w, min_h = w_img * 0.15, h_img * 0.1
    min_area = (w_img * h_img) * 0.01

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > min_w and h > min_h and w * h > min_area and 1.5 < w / h < 3:
            filtered.append((x, y, w, h))
    return filtered

# ============================================================
# 🎯 CENTRAGE ET REDRESSEMENT
# ============================================================

def center_digit_image(binary_img: np.ndarray, target_size=(100, 100)) -> np.ndarray:
    """Centre un chiffre blanc sur fond noir dans une image carrée, sans affiner les traits."""
    if binary_img is None or len(binary_img.shape) != 2:
        return np.zeros(target_size, dtype=np.uint8)

    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return np.zeros(target_size, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary_img[y:y+h, x:x+w]

    if cropped.size == 0:
        return np.zeros(target_size, dtype=np.uint8)

    size = max(target_size)
    centered = np.zeros((size, size), dtype=np.uint8)

    scale = 0.8 * size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    # 🔸 Utiliser INTER_NEAREST pour ne pas lisser les bords
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    x_offset, y_offset = (size - new_w) // 2, (size - new_h) // 2
    centered[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # kernel = np.ones((3, 3), np.uint8)
    # centered = cv2.dilate(centered, kernel, iterations=1)

    return centered



def straighten_carton(carton_img: np.ndarray) -> np.ndarray:
    """Redresse un carton en se basant sur sa grille détectée."""
    h, w = carton_img.shape[:2]
    gray = cv2.cvtColor(carton_img, cv2.COLOR_BGR2GRAY)
    binary = threshold_binary_inv(gray)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 12, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 12))
    hor_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    ver_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

    intersections = cv2.bitwise_and(hor_lines, ver_lines)
    contours, _ = cv2.findContours(intersections, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 4:
        return carton_img

    points = [cv2.boundingRect(c) for c in contours]
    corners = np.array([[x + w_c / 2, y + h_c / 2] for x, y, w_c, h_c in points], dtype=np.float32)
    sum_pts, diff_pts = corners.sum(axis=1), np.diff(corners, axis=1).reshape(-1)

    src_pts = np.array([
        corners[np.argmin(sum_pts)],     # top-left
        corners[np.argmin(diff_pts)],    # top-right
        corners[np.argmax(sum_pts)],     # bottom-right
        corners[np.argmax(diff_pts)]     # bottom-left
    ], dtype=np.float32)

    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(carton_img, M, (w, h))

# ============================================================
# 🧮 DÉTECTION DU BRUIT ("NEIGE TV")
# ============================================================

def is_static_noise(gray_img: np.ndarray, bin_img: np.ndarray | None = None) -> bool:
    """
    Détecte si une image correspond à du bruit aléatoire ("neige TV").
    Analyse statistique et morphologique.
    """
    variance_thresh = 1200.0
    entropy_thresh = 2.5
    edge_ratio_min = 0.08
    cc_small_area = 20
    cc_small_frac = 0.75
    cc_min_components = 150

    gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY) if len(gray_img.shape) == 3 else gray_img.copy()
    coords = cv2.findNonZero(gray)
    if coords is None:
        return False

    x, y, w, h = cv2.boundingRect(coords)
    H, W = gray.shape[:2]
    y1, y2 = max(0, y), min(H, y + h)
    x1, x2 = max(0, x), min(W, x + w)
    if y2 <= y1 or x2 <= x1:
        return False

    roi_gray = gray[y1:y2, x1:x2]
    roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    variance = float(np.var(roi_blur))
    entropy = float(shannon_entropy(roi_blur))
    edges = cv2.Canny(roi_blur, 100, 200)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size

    crit_gray = variance > variance_thresh and entropy > entropy_thresh and edge_ratio > edge_ratio_min

    crit_cc = False
    if bin_img is not None:
        bin_gray = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY) if len(bin_img.shape) == 3 else bin_img
        roi_bin = np.where(bin_gray[y1:y2, x1:x2] > 127, 255, 0).astype(np.uint8)
        num, _, stats, _ = cv2.connectedComponentsWithStats(roi_bin, connectivity=8)

        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            total_white = np.sum(areas)
            if total_white > 0:
                small_frac = np.sum(areas[areas <= cc_small_area]) / total_white
                crit_cc = len(areas) >= cc_min_components and small_frac >= cc_small_frac

    return crit_gray or crit_cc

# ============================================================
# 🧩 EXTRACTION ET CENTRAGE DES CASES
# ============================================================

def extract_and_center_cells(carton_img: np.ndarray, rows=3, cols=9):
    """
    Découpe un carton en cases (3x9), centre les chiffres valides,
    et retourne une grille 3x9 de np.ndarray ou None.
    """
    h, w = carton_img.shape[:2]
    cell_w, cell_h = w / cols, h / rows
    grid = []

    for r in range(rows):
        row = []
        for c in range(cols):
            x1, y1 = int(c * cell_w), int(r * cell_h)
            x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
            cell = carton_img[y1:y2, x1:x2]

            # Rogner les bords
            margin_top, margin_bottom = 14, 14
            margin_left, margin_right = (10, 20) if c == 0 else (8, 14)
            h_cell, w_cell = cell.shape[:2]

            if (h_cell > margin_top + margin_bottom) and (w_cell > margin_left + margin_right):
                cell = cell[margin_top:h_cell - margin_bottom, margin_left:w_cell - margin_right]

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            strong_binary = np.where(clean > 220, 255, 0).astype(np.uint8)

            white_ratio = cv2.countNonZero(strong_binary) / strong_binary.size
            noise = is_static_noise(strong_binary)

            if 0.33 < white_ratio or white_ratio < 0.02 or noise:
                centered = None
            else:
                centered = center_digit_image(strong_binary)

            row.append(centered)
        grid.append(row)
    return grid

# ============================================================
# 🔢 OCR SUR LES CELLULES
# ============================================================

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


def read_cells_ocr(cells_grid, cell_ocr="tmp"):
    """Effectue la lecture OCR d’une grille 3x9 d’images."""
    result = []

    for r, row in enumerate(cells_grid):
        row_values = []
        for c, cell in enumerate(row):
            if cell is None:
                row_values.append(None)
                continue

            halves = [cell] if c == 0 else [
                cell[:, :cell.shape[1] // 2],
                cell[:, cell.shape[1] // 2:]
            ]

            digits = []
            for i, part in enumerate(halves):
                part = center_digit_image(part)
                kernel = np.ones((2, 2), np.uint8)
                part = cv2.dilate(part, kernel, iterations=1)
                prepped = prepare_for_ocr(part, debug=True, debug_name=f"r{r}_c{c}_part{i}")
                text = pytesseract.image_to_string(prepped, config=OCR_CONFIG).strip()
                if text.isdigit():
                    digits.append(text)

            # 🧩 Construction du nombre selon colonne
            number = None
            if digits:
                try:
                    number = int("".join(digits))
                except ValueError:
                    number = None

            # 🔸 Simplification : ajout de la dizaine selon la colonne
            if number is not None:
                # Si chiffre < 10 et colonne > 0 => on complète la dizaine
                if number < 10 and c > 0:
                    number += c * 10
                # Correction du cas col=8 (dernière colonne) → max 90
                if c == 8 and number == 0:
                    number = 90
                

            # --- Debug : sauvegarder image OCRisée ---
            if DEBUG:
                debug_name = f"{cell_ocr}/r{r}_c{c}_part{i}_{text or 'none'}.png"
                os.makedirs(os.path.join(OCR_DEBUG_DIR, cell_ocr), exist_ok=True)
                debug_img = cv2.putText(prepped.copy(), text, (5, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                cv2.imwrite(os.path.join(OCR_DEBUG_DIR, debug_name), debug_img)

            row_values.append(number)
        result.append(row_values)

    return result


# ============================================================
# 🚀 PIPELINE PRINCIPAL
# ============================================================

async def process_bingo_image(image_file: UploadFile):
    """
    Pipeline principal : 
    - Lit une image uploadée
    - Détecte les cartons
    - Extrait, centre et lit les chiffres
    """
    img = read_image_from_bytes(await image_file.read())

    gray = preprocess_gray(img)
    binary = threshold_binary_inv(gray)
    grid_mask = extract_grid_mask(binary)

    contours, _ = cv2.findContours(grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_boxes = filter_contours(contours, *img.shape[1::-1])

    all_results = []

    for idx, (x, y, w, h) in enumerate(filtered_boxes):
        carton_img = straighten_carton(img[y:y+h, x:x+w])
        cells = extract_and_center_cells(carton_img)
        numbers = read_cells_ocr(cells, cell_ocr=f"cell_ocr_{idx+1}")
        all_results.append({"carton": idx + 1, "lines": numbers})

    return {"cartons": all_results}
