"""EJECUCIÓN COMPLETA DE EXTREMO A EXTREMO — congelado de resultados.

    python run_all.py

Reproduce todo el TFM desde las fuentes originales:
  1. Bronze/Silver/Gold de mercado + 6 validaciones de integridad temporal
  2. Experimento 1 (walk-forward, 9 modelos)
  3. Noticias -> FinBERT -> Gold market+sentiment
  4. Experimento 2 (market vs market+sentiment, misma ventana)
  5. Experimento 3 (ensemble dinámico)
  6. Señales BUY/HOLD/SELL, divisa y métricas financieras
  7. Figuras de la memoria

Las descargas y FinBERT se reanudan desde caché: una segunda ejecución es rápida.
Para probar la reproducibilidad desde cero, borrar data/ y outputs/ antes.
"""
import time
from src.ensemble import dynamic, signals
from src.features import build_gold, build_silver
from src.ingestion import download_market, download_news
from src.models import experiment2, train_eval
from src.reporting import make_figures
from src.sentiment import build_news_silver, build_sentiment_gold, finbert_score
from src.validation import checks_phase1

STEPS = [
    ("1/11  BRONZE mercado", download_market),
    ("2/11  SILVER mercado", build_silver),
    ("3/11  GOLD market-only", build_gold),
    ("4/11  VALIDACIÓN de fugas", checks_phase1),
    ("5/11  E1 walk-forward", train_eval),
    ("6/11  BRONZE noticias", download_news),
    ("7/11  SILVER noticias", build_news_silver),
    ("8/11  FinBERT", finbert_score),
    ("9/11  GOLD market+sentiment", build_sentiment_gold),
    ("10/11 E2 + E3 + señales", None),
    ("11/11 Figuras", make_figures),
]

if __name__ == "__main__":
    t0 = time.time()
    for label, mod in STEPS:
        print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
        if mod is None:
            for m in (experiment2, dynamic, signals):
                m.main()
        else:
            mod.main()
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETO EN {(time.time() - t0) / 60:.1f} MIN")
    print(f"  Resultados congelados. No modificar el pipeline a partir de aquí.")
    print(f"{'=' * 70}")
