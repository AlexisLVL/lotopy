import cv2
import numpy as np
from app.services.ocr.ocr_alignment import center_and_crop_digit_image
from app.services.ocr.ocr_noise import is_static_noise

def extract_and_center_cells(carton_img, rows=3, cols=9, name=None):
    h, w = carton_img.shape[:2]
    cell_w, cell_h = w / cols, h / rows
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            x1, y1 = int(c*cell_w), int(r*cell_h)
            x2, y2 = int((c+1)*cell_w), int((r+1)*cell_h)
            cell = carton_img[y1:y2, x1:x2]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            noise = is_static_noise(binary)
            debug_name = None
            if name:
                debug_name = f"r{r}_c{c}_part{name}"
            row.append(None if noise else center_and_crop_digit_image(binary, crop_left=10, crop_right=15, crop_top=25, crop_bottom=50, debug_name=debug_name, context="ocr"))
        grid.append(row)
    return grid
