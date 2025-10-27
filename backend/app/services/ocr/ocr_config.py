import os

OCR_CONFIG = (
    "--oem 1 --psm 13 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c load_system_dawg=false "
    "-c load_freq_dawg=false "
    "-c classify_bln_numeric_mode=1 "
    "-c tessedit_adaption_debug=0 "
)

DEBUG = True
BASE_DIR = "detected_cartons"
OCR_DEBUG_DIR = os.path.join(BASE_DIR, "ocr_debug")

os.makedirs(OCR_DEBUG_DIR, exist_ok=True)
