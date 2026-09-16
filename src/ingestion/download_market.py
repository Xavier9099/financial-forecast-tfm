"""BRONZE — Descarga de series de mercado desde Yahoo Finance (yfinance).

Guarda un parquet por símbolo, tal cual viene de la fuente (sin transformar),
más un manifest.json con la trazabilidad de la descarga.

Uso:  python -m src.ingestion.download_market
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src.config import BRONZE, END, FX, FX_FALLBACK, START, SYMBOLS_ALL, ensure_dirs

RENAME = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance devuelve MultiIndex en algunas versiones incluso con 1 símbolo."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        df.columns = lvl0 if set(RENAME).intersection(lvl0) else df.columns.get_level_values(1)
    return df


def download_symbol(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START, end=END, auto_adjust=False,
                     progress=False, actions=False)
    df = _flatten(df)
    if df.empty and symbol == FX:
        print(f"  [warn] {symbol} vacío, probando fallback {FX_FALLBACK}")
        df = _flatten(yf.download(FX_FALLBACK, start=START, end=END,
                                  auto_adjust=False, progress=False, actions=False))
    if df.empty:
        raise RuntimeError(f"Descarga vacía para {symbol}")

    df = df.rename(columns=RENAME).reset_index().rename(columns={"Date": "date"})
    keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"]
            if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    if "adj_close" not in df:            # índices y FX no traen Adj Close
        df["adj_close"] = df["close"]
    if "volume" not in df:
        df["volume"] = pd.NA
    df["symbol"] = symbol
    df["ingested_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    manifest = {"start": START, "end": END,
                "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": "Yahoo Finance vía yfinance", "symbols": {}}

    for sym in SYMBOLS_ALL:
        print(f"→ descargando {sym} ...")
        df = download_symbol(sym)
        fname = sym.replace("^", "IDX_").replace("=", "_") + ".parquet"
        df.to_parquet(BRONZE / fname, index=False)
        manifest["symbols"][sym] = {
            "file": fname, "rows": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "n_nulls_close": int(df["close"].isna().sum()),
        }
        print(f"   {len(df):>6} filas  {df['date'].min().date()} → {df['date'].max().date()}")

    (BRONZE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nBronze listo en {BRONZE}")


if __name__ == "__main__":
    main()
