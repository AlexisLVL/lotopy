"""
Module : model_predict.py
-------------------------
Responsable de la prédiction de chiffres à partir d’images individuelles
(fond blanc, chiffres noirs), à l’aide d’un modèle CNN Keras et d’un fallback
Tesseract si la confiance est insuffisante.

Chargement paresseux du modèle Keras pour éviter les ralentissements au démarrage.
"""

import os
import numpy as np
import cv2
import pytesseract
from typing import Optional
from threading import Lock

# ============================================================
# 🔧 CONFIGURATION
# ============================================================

MODEL_PATH = os.environ.get("DIGIT_MODEL_PATH", "app/models/digit_cnn.keras")
_CONF_THRESHOLD = 0.60  # confiance minimale du CNN pour accepter une prédiction

_model = None
_model_error = None


# ============================================================
# ⚙️ CHARGEMENT PARESSEUX DU MODÈLE
# ============================================================

_model_lock = Lock()

def get_model():
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return _model

    with _model_lock:
        if _model is None and _model_error is None:
            try:
                from tensorflow.keras.models import load_model
                _model = load_model(MODEL_PATH)
                print(f"[OCR] Modèle chargé depuis {MODEL_PATH}")
            except Exception as e:
                _model_error = e
                print(f"[OCR] ⚠️ Impossible de charger le modèle CNN : {e}")

    return _model



# ============================================================
# 🔢 PREDICTION CNN
# ============================================================

def predict_with_cnn(img: np.ndarray) -> tuple[Optional[int], float]:
    """
    Prédit le chiffre à partir d’une image via le modèle CNN.
    Retourne (chiffre, confiance) ou (None, 0.0) si échec.
    """
    model = get_model()
    if model is None or img is None:
        return None, 0.0

    # Conversion en grayscale + resize + normalisation
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # shape (1, 64, 64, 1)

    preds = model.predict(img, verbose=0)
    digit = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return (digit if confidence >= _CONF_THRESHOLD else None), confidence


# ============================================================
# 🔁 FALLBACK TESSERACT
# ============================================================

def predict_with_tesseract(img: np.ndarray) -> Optional[int]:
    """
    Utilise Tesseract en secours si le CNN échoue ou n’est pas sûr.
    Fonctionne mieux sur des chiffres nets et contrastés.
    """
    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarisation et inversion si nécessaire
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    config = (
        "--oem 1 --psm 10 "
        "-c tessedit_char_whitelist=0123456789 "
        "-c load_system_dawg=false "
        "-c load_freq_dawg=false "
        "-c classify_bln_numeric_mode=1 "
    )

    text = pytesseract.image_to_string(binary, config=config).strip()
    try:
        return int(text) if text.isdigit() else None
    except ValueError:
        return None


# ============================================================
# 🚀 FONCTION PRINCIPALE UTILISÉE PAR L’OCR
# ============================================================

def predict_digit(img: np.ndarray) -> Optional[int]:
    """
    Tente de prédire un chiffre à partir d’une image :
    1. Essaye le modèle CNN
    2. Si la confiance est faible, essaye Tesseract
    """
    digit, conf = predict_with_cnn(img)

    if digit is not None:
        # CNN confiant
        return digit

    # Fallback Tesseract
    digit_tess = predict_with_tesseract(img)
    #if digit_tess is not None:
        #print(f"[OCR] Fallback Tesseract utilisé → {digit_tess}")
    return digit_tess
