import cv2
import numpy as np
from skimage.measure import shannon_entropy

def is_static_noise(gray_img, bin_img=None):
    variance_thresh = 1200.0
    entropy_thresh = 2.5
    edge_ratio_min = 0.08
    gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY) if len(gray_img.shape) == 3 else gray_img
    roi_blur = cv2.GaussianBlur(gray, (3,3), 0)
    variance = float(np.var(roi_blur))
    entropy = float(shannon_entropy(roi_blur))
    edges = cv2.Canny(roi_blur, 100, 200)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size
    return variance > variance_thresh and entropy > entropy_thresh and edge_ratio > edge_ratio_min
