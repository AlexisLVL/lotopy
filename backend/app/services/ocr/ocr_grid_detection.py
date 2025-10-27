import os
import cv2
import numpy as np
from typing import List, Tuple
from app.services.ocr.ocr_config import DEBUG, OCR_DEBUG_DIR

Rect = Tuple[int, int, int, int]  # (x, y, w, h)

def _rect_contains(a: Rect, b: Rect, margin: int = 4) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (bx >= ax + margin and by >= ay + margin and
            bx + bw <= ax + aw - margin and
            by + bh <= ay + ah - margin)

def _grid_density(grid_mask: np.ndarray, rect: Rect) -> float:
    x, y, w, h = rect
    roi = grid_mask[max(0,y):y+h, max(0,x):x+w]
    if roi.size == 0:
        return 0.0
    return float(cv2.countNonZero(roi)) / float(roi.size)

def _aspect_ok(w: int, h: int) -> bool:
    # Carton ~ 3:1 (ajuste si besoin)
    ratio = w / max(1, h)
    return 1.9 <= ratio <= 3.6

def _area_ok(w_img: int, h_img: int, w: int, h: int) -> bool:
    A = w * h
    A_img = w_img * h_img
    return (A >= 0.02 * A_img) and (A <= 0.9 * A_img)

def filter_contours_smart(
    contours,
    hierarchy,
    grid_mask: np.ndarray,
    w_img: int,
    h_img: int,
    base_image: np.ndarray | None = None,
    debug_name: str = "filtered_cartons"
) -> List[Rect]:
    """
    Filtre les contours en éliminant les bords externes quand une grille interne est imbriquée.
    S'appuie sur la hiérarchie + densité de lignes dans grid_mask.
    """
    # 1) Construire les candidats primaires (filtre surf/ratio de base)
    primaries: List[Rect] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not _aspect_ok(w, h): 
            continue
        if not _area_ok(w_img, h_img, w, h):
            continue
        primaries.append((x, y, w, h))

    if not primaries:
        if DEBUG and base_image is not None:
            os.makedirs(OCR_DEBUG_DIR, exist_ok=True)
            dbg = base_image.copy()
            if dbg.ndim == 2:
                dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(OCR_DEBUG_DIR, f"{debug_name}_none.png"), dbg)
        return []

    # 2) Regrouper par imbrication (si A contient B → même groupe)
    groups: List[List[Rect]] = []
    used = [False] * len(primaries)

    for i, a in enumerate(primaries):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        for j, b in enumerate(primaries):
            if i == j or used[j]:
                continue
            if _rect_contains(a, b) or _rect_contains(b, a):
                group.append(b)
                used[j] = True
        groups.append(group)

    # 3) Choisir le meilleur rect par groupe via densité de grille
    selected: List[Rect] = []
    for group in groups:
        best_rect = None
        best_score = -1.0
        for rect in group:
            dens = _grid_density(grid_mask, rect)
            # On peut “pondérer” par la proximité de 3:1 pour départager
            x, y, w, h = rect
            ratio = w / max(1, h)
            ratio_score = 1.0 - min(abs(ratio - 3.0) / 1.5, 1.0)  # score ∈ [0,1]
            score = 0.8 * dens + 0.2 * ratio_score
            if score > best_score:
                best_score = score
                best_rect = rect
        if best_rect:
            selected.append(best_rect)

    # 4) (Option) Éliminer des doublons quasi identiques
    # Tri par aire décroissante et IoU simple sur rectangulaires axis-aligned
    def _iou(a: Rect, b: Rect) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
        inter = max(0, x2-x1) * max(0, y2-y1)
        ua = aw*ah + bw*bh - inter
        return inter/ua if ua > 0 else 0.0

    final: List[Rect] = []
    for rect in sorted(selected, key=lambda r: r[2]*r[3], reverse=True):
        if all(_iou(rect, f) < 0.6 for f in final):
            final.append(rect)

    # 5) Debug visuel
    if DEBUG:
        os.makedirs(OCR_DEBUG_DIR, exist_ok=True)
        if base_image is None:
            dbg = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        else:
            dbg = base_image.copy()
            if dbg.ndim == 2:
                dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)

        for i, (x, y, w, h) in enumerate(final):
            color = tuple(int(c) for c in np.random.randint(80,255,size=3))
            cv2.rectangle(dbg, (x, y), (x+w, y+h), color, 2)
            cv2.putText(dbg, f"C{i+1}", (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Affiche densité pour debug
            dens = _grid_density(grid_mask, (x,y,w,h))
            cv2.putText(dbg, f"d={dens:.2f}", (x+5, y+h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(os.path.join(OCR_DEBUG_DIR, f"{debug_name}.png"), dbg)
        print(f"[OCR] ✅ Cartons retenus: {len(final)} → {os.path.join(OCR_DEBUG_DIR, f'{debug_name}.png')}")

    return final

def extract_grid_mask(binary):
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
    return cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    )

def filter_contours(contours, w_img, h_img, base_image=None, debug_name="filtered_cartons"):
    """
    Filtre les contours selon la taille et le ratio typique d’un carton.
    Si DEBUG est activé, sauvegarde une image avec les contours valides dessinés.
    
    Args:
        contours: liste de contours OpenCV
        w_img, h_img: largeur et hauteur de l’image d’origine
        base_image: image d’origine ou masque sur laquelle dessiner les contours
        debug_name: nom du fichier de sortie pour le debug
    """
    filtered = []
    min_w, min_h = w_img * 0.15, h_img * 0.1
    min_area = (w_img * h_img) * 0.01

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > min_w and h > min_h and w * h > min_area and 1.5 < w / h < 3:
            filtered.append((x, y, w, h))

    # 🔍 Sauvegarde pour debug
    if DEBUG:
        os.makedirs(OCR_DEBUG_DIR, exist_ok=True)

        # Image sur laquelle on dessine les rectangles
        if base_image is None:
            debug_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        else:
            debug_img = base_image.copy()
            if len(debug_img.shape) == 2:
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

        # Couleur et dessin
        for i, (x, y, w, h) in enumerate(filtered):
            color = (0, 255, 0)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(debug_img, f"C{i+1}", (x + 5, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Sauvegarde de l’image
        # save_path = os.path.join(OCR_DEBUG_DIR, f"{debug_name}.png")
        # cv2.imwrite(save_path, debug_img)
        # print(f"[OCR] ✅ Cartons filtrés enregistrés : {save_path} ({len(filtered)} trouvés)")

    return filtered

def is_parent_carton(current_idx: int, boxes: list[tuple[int, int, int, int]], margin: int = 5) -> bool:
    """
    Vérifie si le carton courant (boxes[current_idx]) contient un autre carton.
    Dans ce cas, il s'agit d'un 'carton parent' à ignorer.
    
    Args:
        current_idx: index du carton courant dans la liste boxes
        boxes: liste de tous les cartons détectés [(x, y, w, h), ...]
        margin: marge de tolérance (en pixels) pour le test d'inclusion

    Returns:
        True si le carton courant contient un autre carton (parent), sinon False.
    """
    if not boxes or len(boxes) < 2:
        return False

    x1, y1, w1, h1 = boxes[current_idx]

    for j, rect_b in enumerate(boxes):
        if j == current_idx:
            continue

        xb, yb, wb, hb = rect_b

        if (
            xb >= x1 + margin and
            yb >= y1 + margin and
            xb + wb <= x1 + w1 - margin and
            yb + hb <= y1 + h1 - margin
        ):
            print(f"[OCR] ⚠️ Carton {current_idx+1} contient le carton {j+1}, ignoré.")
            return True

    return False