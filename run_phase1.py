"""Ejecuta la Fase 1 completa de extremo a extremo: Bronze -> Silver -> Gold -> validación.

    python run_phase1.py
"""
from src.ingestion import download_market
from src.features import build_silver, build_gold
from src.validation import checks_phase1

if __name__ == "__main__":
    for step, mod in [("BRONZE", download_market), ("SILVER", build_silver),
                      ("GOLD", build_gold), ("VALIDACIÓN", checks_phase1)]:
        print(f"\n{'=' * 60}\n  {step}\n{'=' * 60}")
        mod.main()
