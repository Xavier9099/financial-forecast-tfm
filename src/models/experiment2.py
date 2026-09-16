"""EXPERIMENTO 2 — ¿aporta el sentimiento capacidad predictiva incremental?

Comparación pareada: los MISMOS modelos, la MISMA ventana (2015-2026) y los MISMOS
pliegues, cambiando únicamente el conjunto de features.

  A) market      -> 29 features de mercado
  B) market+sent -> 29 + 10 de sentimiento

Comparar contra el E1 completo (2011-2026, 15 pliegues) sería inválido: la diferencia
vendría del periodo, no de las variables.

Salidas:
  outputs/predictions/e2_walkforward.parquet
  outputs/metrics/e2_predictive.csv
  outputs/metrics/e2_comparison.csv

Uso:  python -m src.models.experiment2
"""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
from src.backtesting.walkforward import make_folds
from src.config import GOLD, HORIZONS, METRICS, PREDICTIONS, TICKERS, ensure_dirs
from src.features.build_gold import feature_columns
from src.models.metrics import predictive_metrics
from src.models.train_eval import make_models
from src.sentiment.build_sentiment_gold import SENTIMENT_FEATURES

warnings.filterwarnings("ignore")

# Pool congelado en 08_DECISIONS #17
POOL = ["elasticnet", "huber", "pls", "lgbm_reg", "xgb"]

def run_setup(gold: pd.DataFrame, feats: list[str], tag: str) -> pd.DataFrame:
    rows, t0 = [], time.time()
    for ticker in TICKERS:
        g = gold[gold["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        X_all = g[feats].to_numpy(dtype=float)

        for h in HORIZONS:
            y_all = g[f"target_return_{h}d"].to_numpy(dtype=float)
            valid = np.isfinite(y_all)
            X, y = X_all[valid], y_all[valid]
            dates = g.loc[valid, "date"].to_numpy()
            folds = make_folds(int(valid.sum()), h)

            for f in folds:
                Xtr, ytr = X[:f.train_end], y[:f.train_end]
                sl = slice(f.test_start, f.test_end)
                models = {k: v for k, v in make_models().items() if k in POOL}
                for name, model in models.items():
                    model.fit(Xtr, ytr)
                    p = np.asarray(model.predict(X[sl]), dtype=float).ravel()
                    rows.append(pd.DataFrame({
                        "date": dates[sl], "ticker": ticker, "horizon": h,
                        "model": name, "fold": f.idx, "setup": tag,
                        "y_true": y[sl], "y_pred": p,
                    }))
    out = pd.concat(rows, ignore_index=True)
    print(f"  {tag:<12} {len(feats):>2} features | {len(out)} predicciones "
          f"({time.time() - t0:.0f}s)")
    return out

def main() -> None:
    ensure_dirs()
    gold = pd.read_parquet(GOLD / "market_sentiment.parquet")
    market = [c for c in feature_columns(gold) if c not in SENTIMENT_FEATURES]
    both = market + SENTIMENT_FEATURES

    n_folds = len(make_folds(int(np.isfinite(
        gold[gold["ticker"] == TICKERS[0]][f"target_return_5d"]).sum()), 5))
    print(f"Ventana E2: {gold['date'].min():%Y-%m-%d} → {gold['date'].max():%Y-%m-%d} | "
          f"{len(gold)} filas | {n_folds} pliegues por combinación\n")

    pred = pd.concat([run_setup(gold, market, "market"),
                      run_setup(gold, both, "market+sent")], ignore_index=True)
    pred.to_parquet(PREDICTIONS / "e2_walkforward.parquet", index=False)

    # métricas
    rec = []
    for (s, t, h, m), g in pred.groupby(["setup", "ticker", "horizon", "model"]):
        rec.append({"setup": s, "ticker": t, "horizon": h, "model": m,
                    **predictive_metrics(g["y_true"], g["y_pred"])})
    pm = pd.DataFrame(rec)
    pm.to_csv(METRICS / "e2_predictive.csv", index=False)

    # comparación pareada
    piv = pm.pivot_table(index=["ticker", "horizon", "model"], columns="setup",
                         values=["mae", "dir_acc", "spearman_ic"])
    cmp = pd.DataFrame({
        "mae_market": piv[("mae", "market")],
        "mae_sent": piv[("mae", "market+sent")],
        "ic_market": piv[("spearman_ic", "market")],
        "ic_sent": piv[("spearman_ic", "market+sent")],
        "dir_market": piv[("dir_acc", "market")],
        "dir_sent": piv[("dir_acc", "market+sent")],
    })
    cmp["delta_mae"] = cmp["mae_sent"] - cmp["mae_market"]      # negativo = mejora
    cmp["delta_ic"] = cmp["ic_sent"] - cmp["ic_market"]         # positivo = mejora
    cmp["delta_dir"] = cmp["dir_sent"] - cmp["dir_market"]
    cmp = cmp.reset_index()
    cmp.to_csv(METRICS / "e2_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("E2 — EFECTO DEL SENTIMIENTO (Δ = market+sent − market)")
    print("=" * 80)
    print(cmp.pivot_table(index=["ticker", "horizon"], columns="model",
                          values="delta_ic").round(4).to_string())

    print("\nΔ MAE (negativo = el sentimiento mejora)")
    print(cmp.pivot_table(index=["ticker", "horizon"], columns="model",
                          values="delta_mae").round(5).to_string())

    print("\n" + "=" * 80)
    print("VEREDICTO AGREGADO")
    print("=" * 80)
    agg = cmp.groupby("horizon")[["delta_ic", "delta_mae", "delta_dir"]].mean().round(4)
    print(agg.to_string())
    wins = int((cmp["delta_ic"] > 0).sum())
    total = len(cmp)
    print(f"\nCasos en que el sentimiento mejora el IC: {wins}/{total} ({wins/total:.0%})")
    print("Referencia: 50% = indistinguible del azar. Con 6 pliegues la potencia")
    print("estadística es limitada; interpretar como indicio, no como prueba.")
    print(f"\nMétricas guardadas en {METRICS}")

if __name__ == "__main__":
    main()
