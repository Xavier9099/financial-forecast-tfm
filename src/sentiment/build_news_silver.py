"""SILVER (noticias) — Deduplicación, explosión por ticker y fecha efectiva de mercado.

REGLA ANTI-LOOK-AHEAD (el núcleo metodológico del Experimento 2):
  created_at (UTC) -> hora de Nueva York.
  · publicada ANTES de las 16:00 ET de una sesión  -> esa misma sesión
  · publicada DESPUÉS del cierre, o en fin de semana/festivo -> siguiente sesión
Así, el sentimiento de la fila t sólo contiene noticias conocidas al cierre de t,
igual que el resto de features del Gold.

Salida: silver/news.parquet (id, ticker, effective_date, created_at_et, headline, source)

Uso:  python -m src.sentiment.build_news_silver
"""
from __future__ import annotations
import pandas as pd
from src.config import BRONZE, SILVER, TICKERS, ensure_dirs

MARKET_CLOSE_HOUR = 16          # 16:00 ET
TZ = "America/New_York"

def load_bronze() -> pd.DataFrame:
    files = sorted((BRONZE / "news").glob("news_*.parquet"))
    if not files:
        raise SystemExit("No hay noticias en bronze. Ejecuta src.ingestion.download_news")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n0 = len(df)
    # La API filtra por fecha de actualización -> hay artículos repetidos entre años
    df = df.drop_duplicates("id").reset_index(drop=True)
    print(f"Bronze: {n0} filas → {len(df)} titulares únicos ({n0 - len(df)} duplicados)")
    return df

def assign_effective_date(created_utc: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Devuelve la sesión de mercado en la que la noticia ya era conocida al cierre."""
    et = created_utc.dt.tz_convert(TZ)
    day = et.dt.normalize().dt.tz_localize(None)
    after_close = et.dt.hour >= MARKET_CLOSE_HOUR
    # Si llegó después del cierre, el candidato mínimo es el día siguiente
    candidate = day + pd.to_timedelta(after_close.astype(int), unit="D")
    # searchsorted "left": primera sesión >= candidato (salta fines de semana y festivos)
    pos = sessions.searchsorted(candidate.to_numpy(), side="left")
    valid = pos < len(sessions)
    out = pd.Series(pd.NaT, index=created_utc.index, dtype="datetime64[ns]")
    out.loc[valid] = sessions[pos[valid]]
    return out

def main() -> None:
    ensure_dirs()
    news = load_bronze()
    prices = pd.read_parquet(SILVER / "prices.parquet")
    sessions = pd.DatetimeIndex(sorted(prices["date"].unique()))

    news["effective_date"] = assign_effective_date(news["created_at"], sessions)
    n_drop = int(news["effective_date"].isna().sum())
    news = news.dropna(subset=["effective_date"])
    if n_drop:
        print(f"  {n_drop} titulares posteriores a la última sesión: descartados")

    # Un titular menciona varios tickers -> una fila por (titular, ticker) de interés
    news["symbols"] = news["symbols"].fillna("").str.split(",")
    ex = news.explode("symbols").rename(columns={"symbols": "ticker"})
    ex["ticker"] = ex["ticker"].str.strip()
    ex = ex[ex["ticker"].isin(TICKERS)].copy()

    ex["created_at_et"] = ex["created_at"].dt.tz_convert(TZ).dt.tz_localize(None)
    out = (ex[["id", "ticker", "effective_date", "created_at_et", "headline", "source"]]
           .drop_duplicates(["id", "ticker"])
           .sort_values(["ticker", "effective_date"])
           .reset_index(drop=True))

    out.to_parquet(SILVER / "news.parquet", index=False)

    print(f"\nSilver noticias: {len(out)} pares (titular, ticker)")
    for t in TICKERS:
        s = out[out["ticker"] == t]
        cov = s["effective_date"].nunique()
        print(f"  {t:<5} {len(s):>6} titulares | {cov:>5} sesiones con noticias | "
              f"{s['effective_date'].min():%Y-%m-%d} → {s['effective_date'].max():%Y-%m-%d}")

    # Verificación explícita de la regla
    late = ex[ex["created_at_et"].dt.hour >= MARKET_CLOSE_HOUR]
    same_day = (late["effective_date"].dt.normalize()
                == late["created_at_et"].dt.normalize()).sum()
    print(f"\nTitulares tras el cierre asignados al mismo día: {same_day} (debe ser 0)")
    print(f" Titulares únicos a puntuar con FinBERT: {out['id'].nunique()}")

if __name__ == "__main__":
    main()
