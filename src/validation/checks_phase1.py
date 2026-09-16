"""VALIDACIÓN FASE 1 — Integridad temporal y ausencia de look-ahead bias.

Seis comprobaciones; cualquiera que falle detiene el pipeline.

 1. Unicidad e monotonía de (ticker, date).
 2. Sin NaN en la matriz de features tras el warm-up.
 3. Test point-in-time: recalcular las features usando SÓLO datos hasta t debe
    reproducir exactamente la fila t de la tabla Gold. Es la prueba directa de
    causalidad de todo el bloque de feature engineering.
 4. Verificación independiente de los targets por búsqueda directa del precio en t+h.
 5. Consistencia de la identidad de retorno en MXN.
 6. Las últimas h filas de cada ticker deben tener target NaN (no hay futuro aún).

Uso:  python -m src.validation.checks_phase1
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import GOLD, HORIZONS, SEED, SILVER, TICKERS
from src.features.build_gold import asset_features, context_features, feature_columns

RTOL, ATOL = 1e-9, 1e-11
N_PROBES = 12


def _fail(msg: str) -> None:
    raise AssertionError(f" {msg}")

def check_keys(gold: pd.DataFrame) -> None:
    if gold.duplicated(["ticker", "date"]).any():
        _fail("Existen pares (ticker, date) duplicados.")
    for t in TICKERS:
        d = gold.loc[gold["ticker"] == t, "date"]
        if not d.is_monotonic_increasing:
            _fail(f"Fechas no monótonas en {t}.")
    print(" 1. Claves únicas y fechas ordenadas.")

def check_no_nan(gold: pd.DataFrame) -> None:
    feats = feature_columns(gold)
    n = int(gold[feats].isna().sum().sum())
    if n:
        cols = gold[feats].isna().sum()
        _fail(f"{n} NaN en features tras warm-up: {cols[cols > 0].to_dict()}")
    print(f" 2. Sin NaN en {len(feats)} features.")

def check_point_in_time(gold: pd.DataFrame, prices: pd.DataFrame, ctx: pd.DataFrame) -> None:
    """Recalcula features con la serie truncada en t y compara con la tabla Gold."""
    rng = np.random.default_rng(SEED)
    ctx = ctx.sort_values("date").reset_index(drop=True)
    checked = 0

    for t in TICKERS:
        g = prices[prices["ticker"] == t].sort_values("date").reset_index(drop=True)
        gold_t = gold[gold["ticker"] == t].reset_index(drop=True)
        acols = [c for c in asset_features(g.head(300)).columns]
        ccols = [c for c in context_features(ctx.head(300)).columns]

        probes = rng.choice(gold_t.index[50:], size=min(N_PROBES, len(gold_t) - 50),
                            replace=False)
        for i in probes:
            date_t = gold_t.loc[i, "date"]
            g_trunc = g[g["date"] <= date_t].reset_index(drop=True)
            c_trunc = ctx[ctx["date"] <= date_t].reset_index(drop=True)
            a_rec = asset_features(g_trunc).iloc[-1]
            c_rec = context_features(c_trunc).iloc[-1]

            for cols, rec in ((acols, a_rec), (ccols, c_rec)):
                full = gold_t.loc[i, cols].astype(float).to_numpy()
                trunc = rec[cols].astype(float).to_numpy()
                if not np.allclose(full, trunc, rtol=RTOL, atol=ATOL, equal_nan=True):
                    bad = [c for c, a, b in zip(cols, full, trunc)
                           if not np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)]
                    _fail(f"Look-ahead en {t} @ {date_t.date()}: {bad}")
            checked += 1
    print(f"3. Test point-in-time superado ({checked} fechas × "
          f"{len(acols) + len(ccols)} features).")

def check_targets(gold: pd.DataFrame, prices: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED + 1)
    for t in TICKERS:
        g = (prices[prices["ticker"] == t].sort_values("date")
             .reset_index(drop=True))
        pos = {d: k for k, d in enumerate(g["date"])}
        gold_t = gold[gold["ticker"] == t].reset_index(drop=True)
        probes = rng.choice(gold_t.index, size=N_PROBES, replace=False)
        for i in probes:
            k = pos[gold_t.loc[i, "date"]]
            for h in HORIZONS:
                exp = gold_t.loc[i, f"target_return_{h}d"]
                if k + h >= len(g):
                    if not pd.isna(exp):
                        _fail(f"{t}: target_{h}d debería ser NaN al final de la serie.")
                    continue
                real = g.loc[k + h, "adj_close"] / g.loc[k, "adj_close"] - 1.0
                if not np.isclose(exp, real, rtol=1e-9, atol=1e-12):
                    _fail(f"{t} @ {gold_t.loc[i, 'date'].date()} h={h}: "
                          f"target {exp:.8f} ≠ recalculado {real:.8f}")
    print("4. Targets verificados por búsqueda directa en t+h.")

def check_mxn_identity(gold: pd.DataFrame) -> None:
    for h in HORIZONS:
        r, f = gold[f"target_return_{h}d"], gold[f"target_fx_{h}d"]
        expected = (1 + r) * (1 + f) - 1
        diff = (gold[f"target_return_mxn_{h}d"] - expected).abs().max()
        if diff > 1e-12:
            _fail(f"Identidad MXN rota en h={h} (máx {diff:.2e}).")
    print("5. Identidad Return_MXN = (1+r_USD)(1+r_FX)-1 consistente.")

def check_tail_nan(gold: pd.DataFrame) -> None:
    for t in TICKERS:
        sub = gold[gold["ticker"] == t]
        for h in HORIZONS:
            tail = sub[f"target_return_{h}d"].tail(h)
            if tail.notna().any():
                _fail(f"{t}: las últimas {h} filas tienen target_{h}d no nulo.")
    print(" 6. Cola sin targets (no se inventa futuro).")

def main() -> None:
    gold = pd.read_parquet(GOLD / "market_only.parquet")
    prices = pd.read_parquet(SILVER / "prices.parquet")
    ctx = pd.read_parquet(SILVER / "market_context.parquet")

    print(f"\nValidando Gold: {len(gold)} filas, {gold['date'].min().date()} → "
          f"{gold['date'].max().date()}\n")
    check_keys(gold)
    check_no_nan(gold)
    check_point_in_time(gold, prices, ctx)
    check_targets(gold, prices)
    check_mxn_identity(gold)
    check_tail_nan(gold)
    print("\nFASE 1 VALIDADA — la tabla Gold es apta para modelizar.\n")

if __name__ == "__main__":
    main()
