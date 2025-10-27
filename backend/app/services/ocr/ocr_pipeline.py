import cv2
from fastapi import UploadFile
from app.services.ocr.ocr_io import read_image_from_bytes
from app.services.ocr.ocr_preprocessing import preprocess_gray, threshold_binary_inv
from app.services.ocr.ocr_grid_detection import extract_grid_mask, filter_contours, is_parent_carton
from app.services.ocr.ocr_alignment import straighten_carton
from app.services.ocr.ocr_cells import extract_and_center_cells
from app.services.ocr.ocr_recognition import read_cells_ocr, create_carton_comparison_image
from app.services.ocr.ocr_config import DEBUG, OCR_DEBUG_DIR

async def process_bingo_image(image_file: UploadFile):
    img = read_image_from_bytes(await image_file.read())
    gray = preprocess_gray(img)
    binary = threshold_binary_inv(gray)
    grid_mask = extract_grid_mask(binary)
    contours, _ = cv2.findContours(grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = filter_contours(contours, *img.shape[1::-1])
    
    results = []
    for idx, (x, y, w, h) in enumerate(boxes):
        if not is_parent_carton(idx, boxes):
            carton = straighten_carton(img[y:y+h, x:x+w], f"carton_{idx+1}")
            cells = extract_and_center_cells(carton)
            
            ocr_tess = read_cells_ocr(cells, cell_ocr=f"carton_{idx+1}_tess", mode="tesseract")
            ocr_cnn = read_cells_ocr(cells, cell_ocr=f"carton_{idx+1}_cnn", mode="cnn")
            comparison_path = None
            if DEBUG:
                comparison_path = create_carton_comparison_image(
                    carton, ocr_tess, ocr_cnn, debug_name=f"carton_{idx+1}"
                )

            results.append({
                "carton": idx + 1,
                "tesseract": ocr_tess,
                "cnn": ocr_cnn,
                "debug_image": comparison_path
            })
    return {"cartons": results}