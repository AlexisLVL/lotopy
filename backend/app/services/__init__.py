"""
Package des services applicatifs (OCR, Cloudinary, etc.)
"""
from app.services.ocr.ocr_pipeline import process_bingo_image

__all__ = ["process_bingo_image"]
