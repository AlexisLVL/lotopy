from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.ocr.ocr_pipeline import process_bingo_image

router = APIRouter()

@router.post("/bingo")
async def ocr_bingo(image: UploadFile = File(...)):
    """Analyse un carton de bingo via OCR."""
    result = await process_bingo_image(image)
    return JSONResponse(result)
