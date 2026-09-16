"""SILVER — Limpieza, estandarización y alineación temporal.

Reglas de alineación:
  * El calendario maestro es el de SPY (sesiones efectivas del NYSE).
  * VIX y USD/MXN se reindexan a ese calendario con forward-fill (máx. 5 días).
    Forward-fill sólo mira hacia atrás => no introduce look-ahead.
  * Se descartan filas anteriores a la primera fecha con las tres series disponibles.

Salidas:
  silver/prices.parquet          panel largo (date, ticker, OHLCV, adj_close)
  silver/market_context.parquet  (date, spy_adj_close, vix_close, usdmxn)

Uso:  python -m src.features.build_silver
"""
from __future__ import annotations
import pandas as pd
from src.config import BENCHMARK, BRONZE, FX, SILVER, SYMBOLS_EQUITY, TICKERS, VIX, ensure_dirs

FFILL_LIMIT = 5

def _read(symbol: str) -> pd.DataFrame:
    fname = symbol.replace("^", "IDX_").replace("=", "_") + ".parquet"
    df = pd.read_parquet(BRONZE / fname)
    return df.sort_values("date").reset_index(drop=True)

def main() -> None:
    ensure_dirs()

    # Panel de precios
    frames = []
    for sym in SYMBOLS_EQUITY:
        d = _read(sym)[["date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
        d["ticker"] = sym
        frames.append(d)
    prices = pd.concat(frames, ignore_index=True)

    n0 = len(prices)
    prices = prices.dropna(subset=["adj_close", "close"])
    prices = prices[prices["adj_close"] > 0]
    prices = prices.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    print(f"Precios: {n0} → {len(prices)} filas tras limpieza")

    # Calendario maestro
    calendar = (prices.loc[prices["ticker"] == BENCHMARK, "date"]
                .drop_duplicates().sort_values().reset_index(drop=True))
    cal_idx = pd.DatetimeIndex(calendar)

    # Contexto de mercado
    spy = (_read(BENCHMARK).set_index("date")["adj_close"].reindex(cal_idx).rename("spy_adj_close"))
    vix = (_read(VIX).set_index("date")["close"].reindex(cal_idx)
           .ffill(limit=FFILL_LIMIT).rename("vix_close"))
    fx = (_read(FX).set_index("date")["close"].reindex(cal_idx)
          .ffill(limit=FFILL_LIMIT).rename("usdmxn"))

    ctx = pd.concat([spy, vix, fx], axis=1).rename_axis("date").reset_index()
    first_valid = ctx.dropna().iloc[0]["date"]
    ctx = ctx[ctx["date"] >= first_valid].reset_index(drop=True)
    gaps = ctx.isna().sum().to_dict()
    print(f"Contexto: {len(ctx)} filas desde {ctx['date'].min().date()} | NaN restantes: {gaps}")
    ctx = ctx.ffill()  # huecos residuales (festivos MX en FX)

    # Restringimos el panel al mismo calendario y ventana
    prices = prices[prices["date"].isin(set(ctx["date"]))].reset_index(drop=True)

    prices.to_parquet(SILVER / "prices.parquet", index=False)
    ctx.to_parquet(SILVER / "market_context.parquet", index=False)

    for t in TICKERS + [BENCHMARK]:
        sub = prices[prices["ticker"] == t]
        print(f"  {t:<5} {len(sub):>5} filas  {sub['date'].min().date()} → {sub['date'].max().date()}")
    print(f"\nSilver listo en {SILVER}")

if __name__ == "__main__":
    main()
