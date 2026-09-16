"""Fase 3 (datos): noticias -> Silver -> FinBERT -> Gold market+sentiment.

    python run_phase3.py

FinBERT tarda entre 10 y 30 minutos en CPU. Guarda checkpoints: si se corta,
vuelve a lanzarlo y continúa donde iba.
"""
from src.sentiment import build_news_silver, build_sentiment_gold, finbert_score

if __name__ == "__main__":
    for step, mod in [("SILVER NOTICIAS", build_news_silver),
                      ("FINBERT", finbert_score),
                      ("GOLD MARKET+SENTIMENT", build_sentiment_gold)]:
        print(f"\n{'=' * 60}\n  {step}\n{'=' * 60}")
        mod.main()
