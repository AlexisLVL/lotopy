# app/services/model_predict.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Chargement du modèle une seule fois
MODEL_PATH = "app/models/digit_cnn.keras"
model = load_model(MODEL_PATH)

def predict_digit(img: np.ndarray) -> int | None:
    """
    Prédit le chiffre (0–9) à partir d'une image isolée (fond blanc, chiffre noir).
    Retourne None si aucune prédiction fiable.
    """
    if img is None:
        return None

    # Conversion en niveau de gris si besoin
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalisation & redimensionnement
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1, 64, 64, 1)

    # Prédiction
    preds = model.predict(img)
    digit = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Filtrage optionnel (évite les fausses détections)
    return digit if confidence > 0.6 else None
