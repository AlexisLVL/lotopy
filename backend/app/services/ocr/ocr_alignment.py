import os
import cv2
import numpy as np
from app.services.ocr.ocr_config import DEBUG, OCR_DEBUG_DIR

def center_and_crop_digit_image(
    binary_img,
    target_size=(100, 100),
    crop_left=0,
    crop_right=0,
    crop_top=0,
    crop_bottom=0,
    debug_name=None,
    context="cells"
):
    """
    Combine le rognage et le centrage d'une image de chiffre.

    - Étape 1 : Rogne les bords noirs autour du chiffre (crop_digit_image)
    - Étape 2 : Centre le chiffre rogné dans une image carrée (center_digit_image)

    Args:
        binary_img: image binaire (fond noir, texte blanc)
        target_size: taille carrée finale (ex: (100, 100))
        expand_x: expansion horizontale (px ou % si float < 1)
        expand_y: expansion verticale
        debug_name: nom de fichier pour sauvegardes debug
        context: identifiant d'appel (ex: "cells" ou "ocr")

    Returns:
        Image centrée et rognée (np.ndarray)
    """
    cropped = crop_digit_image(
        binary_img,
        crop_left=crop_left,
        crop_right=crop_right,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        debug_name=debug_name,
        context=context
    )

    centered = center_digit_image(
        cropped,
        target_size=target_size,
        debug_name=debug_name,
        context=context
    )

    return centered

# ============================================================
# ✂️ 1️⃣  FONCTION : rogne les bords autour du chiffre
# ============================================================

def crop_digit_image(
    binary_img,
    crop_left=0,
    crop_right=0,
    crop_top=0,
    crop_bottom=0,
    debug_name=None,
    context="cells"
):
    """
    Rogne les bordures noires autour du chiffre (blanc sur fond noir).
    Permet de rogner différemment en haut/bas/gauche/droite.

    Valeurs positives = on coupe plus loin (resserre la zone).
    Valeurs négatives = on garde plus de marge (élargit la zone).

    Args:
        binary_img: image binaire (fond noir, texte blanc)
        crop_left, crop_right, crop_top, crop_bottom: pixels à couper (+) ou ajouter (-)
        debug_name: nom de fichier pour debug
        context: identifiant d'appel ("cells", "ocr", etc.)

    Returns:
        Image rognée (np.ndarray)
    """
    if binary_img is None or len(binary_img.shape) != 2:
        return np.zeros_like(binary_img, dtype=np.uint8)

    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return binary_img

    x, y, w, h = cv2.boundingRect(coords)

    # 🔸 Appliquer les rognages asymétriques
    x1 = min(x + crop_left, binary_img.shape[1])
    x2 = max(x + w - crop_right, 0)
    y1 = min(y + crop_top, binary_img.shape[0])
    y2 = max(y + h - crop_bottom, 0)

    # Sécurité bornes
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(binary_img.shape[1], x2)
    y2 = min(binary_img.shape[0], y2)

    cropped = binary_img[y1:y2, x1:x2]

    # 🔍 Debug
    #if DEBUG and debug_name:
    #    debug_dir = os.path.join(OCR_DEBUG_DIR, f"crop_{context}")
    #    os.makedirs(debug_dir, exist_ok=True)
    #    cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_cropped.png"), cropped)

    return cropped


# ============================================================
# 🎯 2️⃣  FONCTION : centre le chiffre dans une image carrée
# ============================================================

def center_digit_image(binary_img, target_size=(100, 100), debug_name=None, context="cells"):
    """
    Centre une image (déjà rognée) dans une image carrée noire.
    Si DEBUG est actif, sauvegarde avant/après centrage.

    Args:
        binary_img: image binaire (fond noir, texte blanc)
        target_size: taille carrée de sortie (ex: 100x100)
        debug_name: nom du fichier pour debug
        context: identifiant d'appel ("cells" ou "ocr")
    """
    if binary_img is None or len(binary_img.shape) != 2:
        return np.zeros(target_size, dtype=np.uint8)

    h, w = binary_img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros(target_size, dtype=np.uint8)

    size = max(target_size)
    centered = np.zeros((size, size), dtype=np.uint8)

    # Mise à l’échelle proportionnelle
    scale = 0.8 * size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(binary_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    x_off, y_off = (size - new_w) // 2, (size - new_h) // 2
    centered[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # 🔍 Sauvegarde debug
    if DEBUG and debug_name:
        debug_dir = os.path.join(OCR_DEBUG_DIR, f"center_{context}")
        os.makedirs(debug_dir, exist_ok=True)
        debug_visu = centered.copy()
        cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_after.png"), debug_visu)

    return centered

def straighten_carton(carton_img: np.ndarray, debug_name: str = "carton") -> np.ndarray:
    """Redresse un carton en se basant sur sa grille détectée et sauvegarde le résultat si DEBUG est actif."""
    if carton_img is None or carton_img.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    h, w = carton_img.shape[:2]
    gray = cv2.cvtColor(carton_img, cv2.COLOR_BGR2GRAY)
    binary = threshold_binary_inv(gray)

    # 🔹 Détection des lignes horizontales / verticales
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 12, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 12))
    hor_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    ver_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

    intersections = cv2.bitwise_and(hor_lines, ver_lines)
    contours, _ = cv2.findContours(intersections, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 🧩 Si la détection échoue, on garde le carton tel quel
    if len(contours) < 4:
        #if DEBUG:
            #os.makedirs(OCR_DEBUG_DIR, exist_ok=True)
            #cv2.imwrite(os.path.join(OCR_DEBUG_DIR, f"{debug_name}_raw.png"), carton_img)
            #print(f"[OCR] ⚠️ Carton non redressé (moins de 4 coins) → sauvegardé brut.")
        return carton_img

    # 🔸 Calcul des coins du carton
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
    straightened = cv2.warpPerspective(carton_img, M, (w, h))

    # 🔍 Enregistrement du carton redressé pour debug
    # if DEBUG:
    #     os.makedirs(OCR_DEBUG_DIR, exist_ok=True)
    #     save_path = os.path.join(OCR_DEBUG_DIR, f"{debug_name}_straightened.png")
    #     cv2.imwrite(save_path, straightened)
    #     print(f"[OCR] 🧾 Carton redressé enregistré : {save_path}")
    # 
    #     # (Optionnel) Enregistrer aussi une vue avec les contours trouvés
    #     debug_contours = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    #     cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 1)
    #     cv2.imwrite(os.path.join(OCR_DEBUG_DIR, f"{debug_name}_contours.png"), debug_contours)

    return straightened

def threshold_binary_inv(gray: np.ndarray) -> np.ndarray:
    """Applique une binarisation inversée (fond noir, texte blanc)."""
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    return binary
