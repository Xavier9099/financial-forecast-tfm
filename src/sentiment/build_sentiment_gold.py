"""GOLD (market + sentiment) — Agregación diaria del sentimiento y unión con el Gold.

Features de sentimiento, todas causales (usan sólo noticias con effective_date <= t):
  sentiment_mean_1d/3d/7d   polaridad media (p_pos - p_neg) sobre ventanas de sesiones
  positive_news_7d / negative_news_7d
  news_volume_1d / news_volume_7d
  news_volume_z             anomalía de cobertura frente a 63 sesiones
  sentiment_dispersion_7d   desacuerdo entre titulares
  has_news                  1 si hubo noticias ese día

Días sin noticias: polaridad 0 y volumen 0. Es la única opción causal —rellenar con
el valor siguiente sería mirar al futuro— y se documenta como limitación.

Salida: gold/market_sentiment.parquet

Uso:  python -m src.sentiment.build_sentiment_gold
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import GOLD, SILVER, TICKERS, ensure_dirs

SENTIMENT_FEATURES = [
    "sentiment_mean_1d", "sentiment_mean_3d", "sentiment_mean_7d",
    "positive_news_7d", "negative_news_7d",
    "news_volume_1d", "news_volume_7d", "news_volume_z",
    "sentiment_dispersion_7d", "has_news",
]

def daily_sentiment(news: pd.DataFrame, scores: pd.DataFrame,
                    sessions: pd.DatetimeIndex) -> pd.DataFrame:
    df = news.merge(scores, on="id", how="inner")
    df["label"] = df[["p_negative", "p_neutral", "p_positive"]].idxmax(axis=1)

    out = []
    for t in TICKERS:
        s = df[df["ticker"] == t]
        g = s.groupby("effective_date").agg(
            sentiment_mean_1d=("score", "mean"),
            news_volume_1d=("id", "count"),
            pos_1d=("label", lambda x: (x == "p_positive").sum()),
            neg_1d=("label", lambda x: (x == "p_negative").sum()),
            disp_1d=("score", "std"),
        )
        # Rejilla completa de sesiones: los días sin noticias existen y valen 0
        g = g.reindex(sessions).fillna({"sentiment_mean_1d": 0.0, "news_volume_1d": 0,
                                        "pos_1d": 0, "neg_1d": 0, "disp_1d": 0.0})
        g["has_news"] = (g["news_volume_1d"] > 0).astype(float)

        # Medias ponderadas por volumen: un día con 40 titulares pesa más que uno con 1
        wsum = (g["sentiment_mean_1d"] * g["news_volume_1d"])
        for w, name in ((3, "sentiment_mean_3d"), (7, "sentiment_mean_7d")):
            num = wsum.rolling(w, min_periods=1).sum()
            den = g["news_volume_1d"].rolling(w, min_periods=1).sum()
            g[name] = (num / den.replace(0, np.nan)).fillna(0.0)

        g["positive_news_7d"] = g["pos_1d"].rolling(7, min_periods=1).sum()
        g["negative_news_7d"] = g["neg_1d"].rolling(7, min_periods=1).sum()
        g["news_volume_7d"] = g["news_volume_1d"].rolling(7, min_periods=1).sum()
        roll = g["news_volume_1d"].rolling(63, min_periods=21)
        g["news_volume_z"] = ((g["news_volume_1d"] - roll.mean())
                              / roll.std().replace(0, np.nan)).fillna(0.0)
        g["sentiment_dispersion_7d"] = g["disp_1d"].rolling(7, min_periods=1).mean()

        g = g[SENTIMENT_FEATURES].reset_index().rename(columns={"index": "date"})
        g["ticker"] = t
        out.append(g)

    return pd.concat(out, ignore_index=True)

def main() -> None:
    ensure_dirs()
    gold = pd.read_parquet(GOLD / "market_only.parquet")
    news = pd.read_parquet(SILVER / "news.parquet")
    scores = pd.read_parquet(SILVER / "finbert_scores.parquet")
    sessions = pd.DatetimeIndex(sorted(gold["date"].unique()))

    sent = daily_sentiment(news, scores, sessions)
    merged = gold.merge(sent, on=["date", "ticker"], how="left", validate="one_to_one")

    # La ventana con noticias empieza más tarde que la de mercado
    first = news["effective_date"].min()
    merged = merged[merged["date"] >= first].reset_index(drop=True)
    merged[SENTIMENT_FEATURES] = merged[SENTIMENT_FEATURES].fillna(0.0)

    merged.to_parquet(GOLD / "market_sentiment.parquet", index=False)

    print(f"Gold market+sentiment: {len(merged)} filas | "
          f"{merged['date'].min():%Y-%m-%d} → {merged['date'].max():%Y-%m-%d}")
    print(f"NaN: {int(merged[SENTIMENT_FEATURES].isna().sum().sum())}\n")

    cov = merged.groupby("ticker")["has_news"].mean()
    print("Cobertura (proporción de sesiones con al menos un titular):")
    for t, v in cov.items():
        sub = merged[merged["ticker"] == t]
        print(f"  {t:<5} {v:5.1%} | media {sub['news_volume_1d'].mean():4.1f} titulares/sesión"
              f" | polaridad media {sub['sentiment_mean_1d'].mean():+.3f}")

    print("\nCorrelación del sentimiento con el retorno FUTURO (señal bruta):")
    for h in (5, 21, 63):
        c = merged[["sentiment_mean_7d", f"target_return_{h}d"]].corr().iloc[0, 1]
        print(f"  sentiment_mean_7d vs target_{h}d: {c:+.4f}")
    print("\n(Correlaciones de ±0.02 a ±0.05 son lo esperable; valores altos = revisar fugas.)")

if __name__ == "__main__":
    main()
