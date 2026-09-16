"""Métricas predictivas y financieras.

Convenio: `y_true` e `y_pred` son RETORNOS (no precios), de modo que todas las
métricas son comparables entre activos y horizontes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# predictivas
def predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) < 5:
        return {}

    err = y_pred - y_true
    # Directional accuracy: sólo cuenta el signo. El modelo nulo (pred=0) no
    # tiene dirección, así que se marca como NaN en lugar de premiarlo.
    if np.allclose(y_pred, 0.0):
        da = np.nan
    else:
        da = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    sd_p, sd_t = y_pred.std(), y_true.std()
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if sd_p > 1e-12 and sd_t > 1e-12 else np.nan
    ic = (float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman"))
          if sd_p > 1e-12 else np.nan)

    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "dir_acc": da,
        "corr": corr,
        "spearman_ic": ic,
        "bias": float(np.mean(err)),
    }

# financieras
def financial_metrics(rets: np.ndarray, positions: np.ndarray, h: int) -> dict:
    """`rets` son retornos a h sesiones NO solapados; `positions` ∈ {-1, 0, 1}."""
    rets, positions = np.asarray(rets, float), np.asarray(positions, float)
    m = np.isfinite(rets) & np.isfinite(positions)
    rets, positions = rets[m], positions[m]
    if len(rets) < 3:
        return {}

    strat = positions * rets
    equity = np.cumprod(1.0 + strat)
    dd = equity / np.maximum.accumulate(equity) - 1.0
    ppy = 252.0 / h                                   # periodos por año
    sd = strat.std(ddof=1)
    invested = positions != 0

    return {
        "n_periods": int(len(strat)),
        "cum_return": float(equity[-1] - 1.0),
        "ann_return": float(equity[-1] ** (ppy / len(strat)) - 1.0),
        "ann_vol": float(sd * np.sqrt(ppy)),
        "sharpe": float(strat.mean() / sd * np.sqrt(ppy)) if sd > 1e-12 else np.nan,
        "max_drawdown": float(dd.min()),
        "hit_rate": float(np.mean(strat[invested] > 0)) if invested.any() else np.nan,
        "exposure": float(invested.mean()),
    }

def buy_and_hold(rets: np.ndarray, h: int) -> dict:
    return financial_metrics(rets, np.ones_like(rets), h)
