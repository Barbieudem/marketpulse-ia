"""Configuración central del proyecto MarketPulse IA.

Cambia aquí la marca, palabras clave y canales. Todo el pipeline y la app
leen estos valores.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figuras"
for d in (DATA_DIR, OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Si los CSV se subieron junto al código (sin carpeta data), se copian a data/ automáticamente
import shutil
for _nombre in ("google_trends_export.csv", "comentarios.csv"):
    if (ROOT / _nombre).exists() and not (DATA_DIR / _nombre).exists():
        shutil.copy(ROOT / _nombre, DATA_DIR / _nombre)

# --- Marca y mercado -------------------------------------------------------
MARCA = "CeraVe"  # marca real (L'Oréal Dermatological Beauty)
GIRO = "dermocosmética / skincare en México"
# La marca + 4 competidores directos (máx. 5 por consulta de Google Trends = misma escala)
KEYWORDS = ["CeraVe", "La Roche-Posay", "Cetaphil", "Neutrogena", "Nivea"]
KEYWORD_PRINCIPAL = KEYWORDS[0]
GEO = "MX"
TIMEFRAME = "today 5-y"  # 5 años de datos semanales

# --- Canales de inversión (Marketing Mix) ---------------------------------
CANALES = ["Meta Ads", "Google Ads", "TikTok Ads", "Influencers"]
PRESUPUESTO_SEMANAL_TOTAL = 60_000  # MXN
MARGEN_BRUTO = 0.55  # para CLV y ROI
TICKET_PROMEDIO = 380  # MXN

# --- Modelado --------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.25  # último 25% de la serie como prueba (split temporal)
N_LAGS = 4
UMBRAL_FASE = 0.05  # +/-5% de cambio en 4 semanas para Crecimiento/Caída

# --- Gemini ---------------------------------------------------------------
# Modo 1 (AI Studio): GEMINI_API_KEY
# Modo 2 (Vertex AI con créditos GCP): GOOGLE_GENAI_USE_VERTEXAI=true,
#   GOOGLE_CLOUD_PROJECT y GOOGLE_CLOUD_LOCATION
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
