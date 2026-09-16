"""Configuración maestra del proyecto. Única fuente de rutas y parámetros."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BRONZE, SILVER, GOLD = DATA / "bronze", DATA / "silver", DATA / "gold"
OUTPUTS = ROOT / "outputs"
FIGURES, METRICS, PREDICTIONS, MODELS = (
    OUTPUTS / "figures", OUTPUTS / "metrics", OUTPUTS / "predictions", OUTPUTS / "models"
)

# --- Universo congelado del MVP ---
TICKERS = ["AAPL", "MSFT", "JPM"]
BENCHMARK = "SPY"
VIX = "^VIX"
FX = "USDMXN=X"          # USD -> MXN (fallback: "MXN=X")
FX_FALLBACK = "MXN=X"

SYMBOLS_EQUITY = TICKERS + [BENCHMARK]
SYMBOLS_ALL = SYMBOLS_EQUITY + [VIX, FX]

# --- Ventana temporal congelada (reproducibilidad) ---
START = "2005-01-01"
END = "2026-09-11"        # Último cierre incluido. Hardcoreado para reproducibilidad:
                           # la memoria reporta las cifras que este pipeline produce con
                           # esta fecha exacta. Para regenerar con datos más recientes,
                           # cambiar aquí, ejecutar run_all.py completo y revisar todas las cifras nuevamente.
HORIZONS = [5, 21, 63]
SEED = 42

# Mínimo de historia requerido antes de la primera fila utilizable (SMA 200 / máx 252d)
WARMUP_DAYS = 252

def ensure_dirs() -> None:
    for p in (BRONZE, SILVER, GOLD, FIGURES, METRICS, PREDICTIONS, MODELS):
        p.mkdir(parents=True, exist_ok=True)
