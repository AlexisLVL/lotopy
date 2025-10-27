import cv2

def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def threshold_binary_inv(gray):
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    return binary
