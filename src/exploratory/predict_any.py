"""MODO EXPLORATORIO — Predicción para cualquier activo introducido por el usuario.

Replica la metodología del sistema principal sobre un activo arbitrario:
  1. Descarga el histórico y lo alinea al calendario maestro (sesiones de SPY).
  2. Construye las MISMAS 29 variables, con las mismas funciones del pipeline.
  3. Ejecuta un walk-forward reducido (3 pliegues) para estimar la capacidad
     predictiva EN ESE ACTIVO y ponderar los modelos.
  4. Entrena la versión final y predice la última sesión disponible.

DIFERENCIA IMPORTANTE frente a AAPL, MSFT y JPM: aquellos cuentan con 15 pliegues
y 306.180 predicciones fuera de muestra. Aquí hay 3 pliegues. La predicción es
EXPLORATORIA y así debe presentarse. La capacidad predictiva estimada (`est_ic`)
se devuelve precisamente para que el usuario juzgue su fiabilidad.

Uso:  python -m src.exploratory.predict_any NVDA
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from src.backtesting.walkforward import Fold
from src.config import HORIZONS, SILVER, WARMUP_DAYS
from src.ensemble.signals import K_THRESHOLD, MIN_AGREEMENT
from src.features.build_gold import (FEATURE_COLS_EXCLUDE, asset_features,
                                     context_features)
from src.models.metrics import predictive_metrics
from src.models.train_eval import make_models

POOL = ["elasticnet", "huber", "pls", "lgbm_reg", "xgb"]
N_FOLDS, TEST = 3, 252
TAU = 0.05

def build_panel(symbol: str) -> pd.DataFrame:
    """Panel de variables para `symbol`, idéntico en definición al Gold principal."""
    from src.ingestion.download_market import download_symbol

    ctx = pd.read_parquet(SILVER / "market_context.parquet").sort_values("date")
    ctx = ctx.reset_index(drop=True)
    ctxf = pd.concat([ctx, context_features(ctx)], axis=1)

    raw = download_symbol(symbol)
    raw = raw[raw["date"].isin(set(ctx["date"]))].sort_values("date").reset_index(drop=True)
    if len(raw) < WARMUP_DAYS + 2 * TEST:
        raise ValueError(
            f"{symbol}: sólo {len(raw)} sesiones alineadas. Se necesitan al menos "
            f"{WARMUP_DAYS + 2 * TEST} para una validación mínima.")

    g = raw[["date", "close", "adj_close", "volume", "high", "low"]].copy()
    d = pd.concat([g, asset_features(g)], axis=1)
    d = d.merge(ctxf, on="date", how="inner", validate="one_to_one")
    d["excess_ret_21d"] = d["ret_21d"] - d["spy_ret_21d"]
    d["excess_ret_63d"] = d["ret_63d"] - d["spy_ret_63d"]

    p, fx = d["adj_close"].astype(float), d["usdmxn"].astype(float)
    for h in HORIZONS:
        r = p.shift(-h) / p - 1.0
        d[f"target_return_{h}d"] = r
        d[f"target_fx_{h}d"] = fx.shift(-h) / fx - 1.0
        d[f"target_return_mxn_{h}d"] = (1 + r) * (1 + d[f"target_fx_{h}d"]) - 1

    d = d.iloc[WARMUP_DAYS:].reset_index(drop=True)
    d["ticker"] = symbol.upper()
    return d

def _weights(ic: dict[str, float]) -> dict[str, float]:
    v = np.array([ic[m] for m in POOL], dtype=float)
    e = np.exp((v - v.max()) / TAU)
    return dict(zip(POOL, e / e.sum()))

def predict(symbol: str) -> dict:
    panel = build_panel(symbol)
    feats = [c for c in panel.columns
             if c not in FEATURE_COLS_EXCLUDE | {"ticker"} and not c.startswith("target_")]
    X_all = panel[feats].to_numpy(dtype=float)
    out = {"ticker": symbol.upper(), "date": panel["date"].iloc[-1],
           "price": float(panel["adj_close"].iloc[-1]),
           "usdmxn": float(panel["usdmxn"].iloc[-1]),
           "n_sessions": len(panel), "horizons": {}}

    for h in HORIZONS:
        y_all = panel[f"target_return_{h}d"].to_numpy(dtype=float)
        ok = np.isfinite(y_all)
        X, y = X_all[ok], y_all[ok]
        n = len(y)

        #  walk-forward reducido: estima la capacidad predictiva en ESTE activo
        folds, start = [], max(WARMUP_DAYS, n - N_FOLDS * TEST)
        k = 0
        while start + TEST <= n:
            folds.append(Fold(k, start - h, start, start + TEST))
            start += TEST
            k += 1
        folds = folds[-N_FOLDS:]

        ic = {m: 0.0 for m in POOL}
        if folds:
            preds = {m: ([], []) for m in POOL}
            for f in folds:
                models = {k_: v for k_, v in make_models().items() if k_ in POOL}
                for name, model in models.items():
                    model.fit(X[:f.train_end], y[:f.train_end])
                    p = np.asarray(model.predict(X[f.test_start:f.test_end]),
                                   dtype=float).ravel()
                    preds[name][0].append(p)
                    preds[name][1].append(y[f.test_start:f.test_end])
            for m in POOL:
                met = predictive_metrics(np.concatenate(preds[m][1]),
                                         np.concatenate(preds[m][0]))
                ic[m] = float(met.get("spearman_ic") or 0.0)

        w = _weights(ic)

        # modelo final sobre toda la historia utilizable 
        x_last = X_all[-1:].astype(float)
        individual = {}
        for name, model in make_models().items():
            if name not in POOL:
                continue
            model.fit(X, y)
            individual[name] = float(np.asarray(model.predict(x_last)).ravel()[0])

        pred = sum(w[m] * individual[m] for m in POOL)
        vol = float(panel["vol_21d"].iloc[-1])
        thr = K_THRESHOLD * vol * np.sqrt(h)
        agree = float(np.mean([np.sign(individual[m]) == np.sign(pred) for m in POOL]))

        signal = "HOLD"
        if agree >= MIN_AGREEMENT and pred > thr:
            signal = "BUY"
        elif agree >= MIN_AGREEMENT and pred < -thr:
            signal = "SELL"

        out["horizons"][h] = {
            "expected_return": pred, "expected_price": out["price"] * (1 + pred),
            "threshold": thr, "agreement": agree, "signal": signal,
            "weights": w, "est_ic": ic, "best_est_ic": max(ic.values()),
            "n_folds": len(folds), "individual": individual,
        }
    return out

def main() -> None:
    sym = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()
    r = predict(sym)
    print(f"\n{r['ticker']} · {r['date']:%d/%m/%Y} · ${r['price']:,.2f} · "
          f"{r['n_sessions']} sesiones utilizables\n")
    print(f"{'h':>4} {'señal':>6} {'ret.esp.':>10} {'precio esp.':>13} "
          f"{'umbral':>9} {'acuerdo':>8} {'IC est.':>9} {'pliegues':>9}")
    for h, d in r["horizons"].items():
        print(f"{h:>4} {d['signal']:>6} {d['expected_return']:>+9.2%} "
              f"${d['expected_price']:>12,.2f} {d['threshold']:>8.2%} "
              f"{d['agreement']:>7.0%} {d['best_est_ic']:>+8.3f} {d['n_folds']:>9}")
    print("\n Modo exploratorio: validación de 3 pliegues frente a los 15 de los")
    print("  activos del estudio. Interpretar como orientativo. Un IC estimado")
    print("  negativo indica que el sistema NO funciona en este activo.")

if __name__ == "__main__":
    main()
