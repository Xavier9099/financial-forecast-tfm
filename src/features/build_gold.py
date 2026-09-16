"""GOLD — Tabla de features market-only + targets de retorno futuro.

Regla principal: TODA feature en la fila de fecha t usa exclusivamente información
disponible al cierre de t. Todo lo que mira al futuro vive en columnas `target_*`.

Las funciones `asset_features` y `context_features` son puras y causales: recalcularlas
sobre la serie truncada en t devuelve exactamente el mismo valor en t. Esa propiedad
es lo que verifica src/validation/checks_phase1.py.

Uso:  python -m src.features.build_gold
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import GOLD, HORIZONS, SILVER, TICKERS, WARMUP_DAYS, ensure_dirs

# Utilidades causales

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder vía EWM recursivo (adjust=False) => estrictamente causal."""
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def asset_features(g: pd.DataFrame) -> pd.DataFrame:
    """Features del activo. `g` ordenado por fecha, columnas: adj_close, high, low, volume."""
    p = g["adj_close"].astype(float)
    out = pd.DataFrame(index=g.index)

    # Retornos pasados
    for h in (1, 5, 21, 63):
        out[f"ret_{h}d"] = p.pct_change(h)

    # Volatilidad realizada y su régimen
    out["vol_21d"] = out["ret_1d"].rolling(21).std()
    out["vol_63d"] = out["ret_1d"].rolling(63).std()
    out["vol_ratio_21_63"] = out["vol_21d"] / out["vol_63d"]

    # Posición relativa a medias móviles (tendencia, escala-invariante)
    for w in (20, 50, 200):
        out[f"px_vs_sma{w}"] = p / p.rolling(w).mean() - 1.0

    # Momentum y reversión
    out["mom_21_63"] = out["ret_21d"] - out["ret_63d"]
    out["rsi_14"] = _rsi(p)

    # Actividad y rango
    v = g["volume"].astype(float).replace(0.0, np.nan)
    out["volchg_5_63"] = v.rolling(5).mean() / v.rolling(63).mean() - 1.0
    out["range_21d"] = g["high"].rolling(21).max() / g["low"].rolling(21).min() - 1.0
    out["dist_52w_high"] = p / p.rolling(252).max() - 1.0

    return out

def context_features(ctx: pd.DataFrame) -> pd.DataFrame:
    """Features de mercado (SPY), volatilidad implícita (VIX) y divisa (USD/MXN)."""
    out = pd.DataFrame(index=ctx.index)
    spy = ctx["spy_adj_close"].astype(float)
    vix = ctx["vix_close"].astype(float)
    fx = ctx["usdmxn"].astype(float)

    for h in (1, 5, 21, 63):
        out[f"spy_ret_{h}d"] = spy.pct_change(h)
    out["spy_vol_21d"] = out["spy_ret_1d"].rolling(21).std()

    out["vix_level"] = vix
    out["vix_chg_5d"] = vix / vix.shift(5) - 1.0
    out["vix_z_252"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()

    for h in (1, 5, 21):
        out[f"fx_ret_{h}d"] = fx.pct_change(h)
    out["fx_vol_21d"] = out["fx_ret_1d"].rolling(21).std()

    return out

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Targets de retorno futuro en USD, de la divisa y en MXN. Sólo aquí se mira al futuro."""
    p = df["adj_close"].astype(float)
    fx = df["usdmxn"].astype(float)
    for h in HORIZONS:
        r_usd = p.shift(-h) / p - 1.0
        r_fx = fx.shift(-h) / fx - 1.0
        df[f"target_return_{h}d"] = r_usd
        df[f"target_fx_{h}d"] = r_fx
        df[f"target_return_mxn_{h}d"] = (1.0 + r_usd) * (1.0 + r_fx) - 1.0
        df[f"target_dir_{h}d"] = np.where(r_usd.isna(), np.nan, (r_usd > 0).astype(float))
    return df

# Construcción

def build(prices: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    ctx = ctx.sort_values("date").reset_index(drop=True)
    ctxf = pd.concat([ctx, context_features(ctx)], axis=1)

    rows = []
    for t in TICKERS:
        g = (prices[prices["ticker"] == t]
             .sort_values("date").reset_index(drop=True))
        feats = asset_features(g)
        d = pd.concat([g[["date", "ticker", "close", "adj_close", "volume"]], feats], axis=1)
        d = d.merge(ctxf, on="date", how="inner", validate="one_to_one")
        # Fuerza relativa frente al mercado (requiere el merge hecho)
        d["excess_ret_21d"] = d["ret_21d"] - d["spy_ret_21d"]
        d["excess_ret_63d"] = d["ret_63d"] - d["spy_ret_63d"]
        d = add_targets(d)
        rows.append(d)

    gold = pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"])

    # Descartamos el periodo de calentamiento (ventanas largas aún incompletas)
    gold = gold[gold.groupby("ticker").cumcount() >= WARMUP_DAYS].reset_index(drop=True)
    return gold

FEATURE_COLS_EXCLUDE = {"date", "ticker", "close", "adj_close", "volume",
                        "spy_adj_close", "vix_close", "usdmxn"}

def feature_columns(gold: pd.DataFrame) -> list[str]:
    return [c for c in gold.columns
            if c not in FEATURE_COLS_EXCLUDE and not c.startswith("target_")]

def main() -> None:
    ensure_dirs()
    prices = pd.read_parquet(SILVER / "prices.parquet")
    ctx = pd.read_parquet(SILVER / "market_context.parquet")
    gold = build(prices, ctx)

    feats = feature_columns(gold)
    gold.to_parquet(GOLD / "market_only.parquet", index=False)
    pd.Series(feats).to_csv(GOLD / "feature_list.csv", index=False, header=["feature"])

    print(f"Gold: {len(gold)} filas | {len(feats)} features | "
          f"{gold['date'].min().date()} → {gold['date'].max().date()}")
    print(f"NaN en features: {int(gold[feats].isna().sum().sum())}")
    for t in TICKERS:
        sub = gold[gold["ticker"] == t]
        print(f"  {t:<5} {len(sub):>5} filas")
    print(f"\nGold listo en {GOLD / 'market_only.parquet'}")

if __name__ == "__main__":
    main()
