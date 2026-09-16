"""EXPERIMENTO 1 — Modelos market-only evaluados con walk-forward purgado.

Modelos: zero-return, media histórica, Ridge, LightGBM, XGBoost.
Para cada ticker × horizonte × pliegue se entrena, se predice el bloque de test
y se guardan las predicciones fuera de muestra.

Salidas:
  outputs/predictions/e1_walkforward.parquet
  outputs/metrics/e1_predictive.csv
  outputs/metrics/e1_financial.csv

Uso:  python -m src.models.train_eval
"""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.backtesting.walkforward import describe, make_folds
from src.config import GOLD, HORIZONS, METRICS, PREDICTIONS, SEED, TICKERS, ensure_dirs
from src.features.build_gold import feature_columns
from src.models.metrics import buy_and_hold, financial_metrics, predictive_metrics

warnings.filterwarnings("ignore")

# Umbral de señal: |retorno esperado| debe superar este múltiplo de la volatilidad
# realizada a 21d escalada al horizonte. Se documenta en 08_DECISIONS.md.
SIGNAL_K = 0.25

def make_models(seed: int = SEED) -> dict:
    import lightgbm as lgb
    import xgboost as xgb
    return {
        # lineales / baja capacidad: es donde apareció la señal en E1
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed)),
        "elasticnet": make_pipeline(StandardScaler(),
            ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=seed)),
        "huber": make_pipeline(StandardScaler(),
            HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500)),
        "pls": make_pipeline(StandardScaler(), PLSRegression(n_components=3)),
        # árboles
        "lgbm": lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
            n_jobs=1, verbose=-1, deterministic=True, force_row_wise=True),
        "lgbm_reg": lgb.LGBMRegressor(   # misma familia, capacidad muy reducida
            n_estimators=400, learning_rate=0.01, num_leaves=7, max_depth=3,
            min_child_samples=100, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.6, reg_lambda=10.0, random_state=seed,
            n_jobs=1, verbose=-1, deterministic=True, force_row_wise=True),
        "xgb": xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, n_jobs=1,
            tree_method="hist", verbosity=0),
    }

def run() -> pd.DataFrame:
    ensure_dirs()
    gold = pd.read_parquet(GOLD / "market_only.parquet")
    feats = feature_columns(gold)
    print(f"Gold: {len(gold)} filas | {len(feats)} features\n")

    rows, t0 = [], time.time()
    for ticker in TICKERS:
        g = gold[gold["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        X_all = g[feats].to_numpy(dtype=float)

        for h in HORIZONS:
            y_all = g[f"target_return_{h}d"].to_numpy(dtype=float)
            valid = np.isfinite(y_all)                  # descarta la cola sin futuro
            n = int(valid.sum())
            folds = make_folds(n, h)
            if ticker == TICKERS[0] and h == HORIZONS[0]:
                print(f"Pliegues (ejemplo {ticker} h={h}, n={n}):")
                print(describe(folds, g.loc[valid, "date"].dt.date.to_numpy()), "\n")

            X, y = X_all[valid], y_all[valid]
            dates = g.loc[valid, "date"].to_numpy()
            # volatilidad realizada escalada al horizonte -> umbral de señal
            vol_h = g.loc[valid, "vol_21d"].to_numpy() * np.sqrt(h)

            for f in folds:
                Xtr, ytr = X[:f.train_end], y[:f.train_end]
                Xte = X[f.test_start:f.test_end]
                sl = slice(f.test_start, f.test_end)

                preds = {
                    "zero": np.zeros(len(Xte)),
                    "hist_mean": np.full(len(Xte), ytr.mean()),
                }
                for name, model in make_models().items():
                    model.fit(Xtr, ytr)
                    # PLS devuelve (n,1); el resto (n,) -> se normaliza la forma
                    preds[name] = np.asarray(model.predict(Xte), dtype=float).ravel()

                for name, p in preds.items():
                    rows.append(pd.DataFrame({
                        "date": dates[sl], "ticker": ticker, "horizon": h,
                        "model": name, "fold": f.idx,
                        "y_true": y[sl], "y_pred": p, "vol_h": vol_h[sl],
                    }))

            print(f"  {ticker} h={h:<3} {len(folds)} pliegues  ({time.time() - t0:.0f}s)")

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(PREDICTIONS / "e1_walkforward.parquet", index=False)
    print(f"\nPredicciones fuera de muestra: {len(out)} filas → e1_walkforward.parquet")
    return out

def evaluate(pred: pd.DataFrame) -> None:
    # métricas predictivas
    rec = []
    for (t, h, m), g in pred.groupby(["ticker", "horizon", "model"]):
        rec.append({"ticker": t, "horizon": h, "model": m,
                    **predictive_metrics(g["y_true"], g["y_pred"])})
    pm = pd.DataFrame(rec)
    # Tasa base: % de retornos positivos. Es el listón real de la directional
    # accuracy; sin ella, un modelo que siempre predice "sube" parece brillante.
    base_rate = (pred[pred["model"] == "zero"].groupby(["ticker", "horizon"])["y_true"]
                 .apply(lambda s: float((s > 0).mean())).rename("base_rate"))
    pm = pm.merge(base_rate, on=["ticker", "horizon"], how="left")
    pm["dir_acc_vs_base"] = pm["dir_acc"] - pm["base_rate"]
    pm = pm.sort_values(["ticker", "horizon", "mae"])
    pm.to_csv(METRICS / "e1_predictive.csv", index=False)

    # métricas financieras (sin solapamiento)
    rec = []
    for (t, h, m), g in pred.groupby(["ticker", "horizon", "model"]):
        g = g.sort_values("date").iloc[::h]            # 1 observación cada h sesiones
        thr = SIGNAL_K * g["vol_h"].to_numpy()
        pos = np.where(g["y_pred"] > thr, 1.0, np.where(g["y_pred"] < -thr, -1.0, 0.0))
        fm = financial_metrics(g["y_true"].to_numpy(), pos, h)
        if fm:
            rec.append({"ticker": t, "horizon": h, "model": m, **fm})
        if m == "lgbm":                                 # referencia buy & hold
            bh = buy_and_hold(g["y_true"].to_numpy(), h)
            rec.append({"ticker": t, "horizon": h, "model": "buy_and_hold", **bh})
    fmdf = pd.DataFrame(rec).sort_values(["ticker", "horizon", "sharpe"], ascending=[1, 1, 0])
    fmdf.to_csv(METRICS / "e1_financial.csv", index=False)

    # resumen en consola
    print("\n" + "=" * 70)
    print("RESUMEN PREDICTIVO (media entre tickers)")
    print("=" * 70)
    summ = (pm.groupby(["horizon", "model"])[
                ["mae", "rmse", "dir_acc", "base_rate", "dir_acc_vs_base", "spearman_ic"]]
            .mean().round(4))
    print(summ.to_string())

    print("\n" + "=" * 70)
    print("IC DE SPEARMAN POR TICKER (la métrica que importa para una señal)")
    print("=" * 70)
    print(pm.pivot_table(index=["ticker", "horizon"], columns="model",
                         values="spearman_ic").round(4).to_string())

    print("\n" + "=" * 70)
    print("¿BATE AL MODELO NULO? (MAE modelo / MAE zero-return, <1 es mejor)")
    print("=" * 70)
    base = pm[pm["model"] == "zero"].set_index(["ticker", "horizon"])["mae"]
    rel = pm.copy()
    rel["mae_rel"] = rel.apply(lambda r: r["mae"] / base.loc[(r["ticker"], r["horizon"])], axis=1)
    print(rel.pivot_table(index=["ticker", "horizon"], columns="model",
                          values="mae_rel").round(4).to_string())

    print("\n" + "=" * 70)
    print("SHARPE ANUALIZADO POR ESTRATEGIA")
    print("=" * 70)
    print(fmdf.pivot_table(index=["ticker", "horizon"], columns="model",
                           values="sharpe").round(2).to_string())
    print(f"\nMétricas guardadas en {METRICS}")

def main() -> None:
    evaluate(run())

if __name__ == "__main__":
    main()
