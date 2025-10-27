import cv2
import numpy as np

def read_image_from_bytes(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def save_image(img, path: str):
    cv2.imwrite(path, img)
