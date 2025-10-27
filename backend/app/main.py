from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import ocr_routes, healthcheck
import os
app = FastAPI(title="Bingo OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr_routes.router, prefix="/api/ocr", tags=["OCR"])
app.include_router(healthcheck.router, prefix="/api", tags=["Healthcheck"])

@app.get("/")
def root():
    return {"message": "Bienvenue sur l’API Bingo OCR 🎯"}

# Servir les fichiers de debug
if os.path.exists("detected_cartons"):
    app.mount("/detected_cartons", StaticFiles(directory="detected_cartons"), name="detected_cartons")