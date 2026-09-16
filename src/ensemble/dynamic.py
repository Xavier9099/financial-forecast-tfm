"""EXPERIMENTO 3 — Ensemble dinámico con pesos point-in-time.

Para cada ticker × horizonte × pliegue k, los pesos se estiman ÚNICAMENTE con el
desempeño observado en pliegues anteriores a k. El pliegue 0 no tiene historia, así
que arranca con pesos iguales. Ningún peso usa información del periodo que predice.

No se reentrena ningún modelo: se combinan predicciones ya generadas fuera de muestra
(`e1_walkforward.parquet`, `e2_walkforward.parquet`). Eso hace el cálculo instantáneo
y trivialmente auditable.

Esquemas de ponderación
  equal        media simple (referencia obligatoria: si el ensemble dinámico no la
               bate, la adaptación no aporta nada)
  inv_mae      w ∝ (1/MAE)^λ
  mae_da       w ∝ softmax(½·z(−MAE) + ½·z(DirAcc))
  softmax_ic   w ∝ softmax(IC de Spearman / τ)
  best_recent  selección dura del mejor modelo reciente (w = 1)

Ventanas
  expanding    todos los pliegues anteriores
  last3        sólo los 3 pliegues anteriores (adaptación rápida)

Salidas:
  outputs/predictions/e3_ensemble.parquet
  outputs/metrics/e3_predictive.csv
  outputs/metrics/e3_weights.csv      <- las usa la demo de Streamlit

Uso:  python -m src.ensemble.dynamic
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import METRICS, PREDICTIONS, ensure_dirs
from src.models.metrics import predictive_metrics

LAMBDA = 4.0      # discriminación del inverse-MAE
TAU = 0.05        # temperatura del softmax sobre IC
SCHEMES = ["equal", "inv_mae", "mae_da", "softmax_ic", "best_recent"]
WINDOWS = {"expanding": None, "last3": 3}

def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)

def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    e = np.exp((x - x.max()) / temp)
    return e / e.sum()

def fold_scores(hist: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """MAE, acierto direccional e IC de cada candidato sobre los pliegues de historia."""
    rec = {}
    for m in models:
        g = hist[hist["model"] == m]
        if len(g) < 20:
            rec[m] = {"mae": np.nan, "da": np.nan, "ic": np.nan}
            continue
        yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        ic = pd.Series(yp).corr(pd.Series(yt), method="spearman")
        rec[m] = {"mae": float(np.mean(np.abs(yp - yt))),
                  "da": float(np.mean(np.sign(yp) == np.sign(yt))),
                  "ic": 0.0 if pd.isna(ic) else float(ic)}
    return pd.DataFrame(rec).T

def weights_from(scores: pd.DataFrame, scheme: str) -> np.ndarray:
    models = scores.index.to_numpy()
    n = len(models)
    if scores["mae"].isna().all() or scheme == "equal":
        return np.full(n, 1.0 / n)

    mae = scores["mae"].fillna(scores["mae"].max()).to_numpy()
    da = scores["da"].fillna(0.5).to_numpy()
    ic = scores["ic"].fillna(0.0).to_numpy()

    if scheme == "inv_mae":
        w = (1.0 / np.maximum(mae, 1e-9)) ** LAMBDA
        return w / w.sum()
    if scheme == "mae_da":
        return _softmax(0.5 * _z(-mae) + 0.5 * _z(da), temp=0.5)
    if scheme == "softmax_ic":
        return _softmax(ic, temp=TAU)
    if scheme == "best_recent":
        w = np.zeros(n)
        w[int(np.argmax(ic))] = 1.0
        return w
    raise ValueError(scheme)

def build(pred: pd.DataFrame, models: list[str], tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ens_rows, wrows = [], []

    for (ticker, h), g in pred.groupby(["ticker", "horizon"]):
        folds = sorted(g["fold"].unique())
        for scheme in SCHEMES:
            for wname, wsize in WINDOWS.items():
                if scheme == "equal" and wname != "expanding":
                    continue                      # no depende de la ventana
                for k in folds:
                    prior = [f for f in folds if f < k]
                    if wsize:
                        prior = prior[-wsize:]
                    cur = g[g["fold"] == k]
                    piv = cur.pivot_table(index="date", columns="model", values="y_pred")
                    piv = piv.reindex(columns=models)
                    if piv.isna().all(axis=None):
                        continue

                    if not prior:                 # pliegue 0: sin historia
                        w = np.full(len(models), 1.0 / len(models))
                        sc = pd.DataFrame(index=models)
                    else:
                        sc = fold_scores(g[g["fold"].isin(prior)], models)
                        w = weights_from(sc, scheme)

                    yhat = np.nansum(piv.to_numpy() * w, axis=1)
                    truth = (cur[cur["model"] == models[0]]
                             .set_index("date")["y_true"].reindex(piv.index))
                    ens_rows.append(pd.DataFrame({
                        "date": piv.index, "ticker": ticker, "horizon": h,
                        "fold": k, "model": f"ens_{scheme}_{wname}",
                        "y_true": truth.to_numpy(), "y_pred": yhat}))
                    for m, wi in zip(models, w):
                        wrows.append({"source": tag, "ticker": ticker, "horizon": h,
                                      "fold": k, "scheme": scheme, "window": wname,
                                      "model": m, "weight": float(wi)})

    return pd.concat(ens_rows, ignore_index=True), pd.DataFrame(wrows)

def evaluate(pred: pd.DataFrame, ens: pd.DataFrame, tag: str) -> pd.DataFrame:
    allp = pd.concat([pred[["date", "ticker", "horizon", "model", "y_true", "y_pred"]],
                      ens[["date", "ticker", "horizon", "model", "y_true", "y_pred"]]],
                     ignore_index=True)
    rec = []
    for (t, h, m), g in allp.groupby(["ticker", "horizon", "model"]):
        rec.append({"source": tag, "ticker": t, "horizon": h, "model": m,
                    **predictive_metrics(g["y_true"], g["y_pred"])})
    return pd.DataFrame(rec)

def report(pm: pd.DataFrame, models: list[str], tag: str) -> None:
    print(f"\n{'=' * 80}\n{tag} — IC DE SPEARMAN MEDIO ENTRE TICKERS\n{'=' * 80}")
    summ = (pm.groupby(["horizon", "model"])["spearman_ic"].mean()
            .unstack(0).round(4).sort_values(by=pm["horizon"].max(), ascending=False))
    print(summ.to_string())

    ens_names = [m for m in pm["model"].unique() if m.startswith("ens_")]
    best_ind = pm[pm["model"].isin(models)].groupby("model")["spearman_ic"].mean().max()
    print(f"\nMejor modelo individual (IC medio global): {best_ind:+.4f}")
    for e in sorted(ens_names):
        v = pm[pm["model"] == e]["spearman_ic"].mean()
        mark = "ok" if v > best_ind else "  "
        print(f"  {mark} {e:<26} {v:+.4f}  ({v - best_ind:+.4f} vs mejor individual)")

    print(f"\n{tag} — ¿gana el ensemble en cada par ticker × horizonte?")
    wide = pm.pivot_table(index=["ticker", "horizon"], columns="model", values="spearman_ic")
    ind_max = wide[[m for m in models if m in wide.columns]].max(axis=1)
    for e in sorted(ens_names):
        wins = int((wide[e] > ind_max).sum())
        print(f"  {e:<26} {wins}/{len(wide)} casos por encima del mejor individual")

def main() -> None:
    ensure_dirs()
    out_pm, out_w, out_ens = [], [], []

    # E3a: market-only, 15 pliegues
    p1 = pd.read_parquet(PREDICTIONS / "e1_walkforward.parquet")
    pool1 = ["elasticnet", "huber", "pls", "lgbm_reg", "xgb"]
    p1 = p1[p1["model"].isin(pool1)].copy()
    print(f"E3a market-only: {len(p1)} predicciones, {p1['fold'].nunique()} pliegues, "
          f"{len(pool1)} candidatos")
    e1, w1 = build(p1, pool1, "e1")
    pm1 = evaluate(p1, e1, "e1")
    report(pm1, pool1, "E3a — MARKET-ONLY (2011-2026, 15 pliegues)")
    out_pm.append(pm1); out_w.append(w1); out_ens.append(e1.assign(source="e1"))

    # E3b: market-only + market+sentiment como candidatos
    p2 = pd.read_parquet(PREDICTIONS / "e2_walkforward.parquet")
    p2["model"] = p2["model"] + "@" + p2["setup"].str.replace("market+sent", "sent")
    pool2 = sorted(p2["model"].unique())
    print(f"\nE3b mixto: {len(p2)} predicciones, {p2['fold'].nunique()} pliegues, "
          f"{len(pool2)} candidatos (5 modelos × 2 conjuntos de features)")
    e2, w2 = build(p2, pool2, "e2")
    pm2 = evaluate(p2, e2, "e2")
    report(pm2, pool2, "E3b — MIXTO market / market+sentiment (2015-2026, 6 pliegues)")
    out_pm.append(pm2); out_w.append(w2); out_ens.append(e2.assign(source="e2"))

    pd.concat(out_pm, ignore_index=True).to_csv(METRICS / "e3_predictive.csv", index=False)
    wdf = pd.concat(out_w, ignore_index=True)
    wdf.to_csv(METRICS / "e3_weights.csv", index=False)
    pd.concat(out_ens, ignore_index=True).to_parquet(
        PREDICTIONS / "e3_ensemble.parquet", index=False)

    # ¿elige el ensemble el sentimiento donde conviene?
    print(f"\n{'=' * 80}\n¿DECIDE EL ENSEMBLE POR SÍ MISMO USAR SENTIMIENTO?\n{'=' * 80}")
    sel = wdf[(wdf["source"] == "e2") & (wdf["scheme"] == "softmax_ic")
              & (wdf["window"] == "expanding") & (wdf["fold"] > 0)].copy()
    sel["usa_sent"] = sel["model"].str.endswith("@sent")
    share = (sel.groupby(["ticker", "horizon"])
             .apply(lambda d: (d.loc[d["usa_sent"], "weight"].sum()
                               / d["weight"].sum()), include_groups=False)
             .unstack(1).round(3))
    print("Peso total asignado a los candidatos CON sentimiento:")
    print(share.to_string())
    print("\n(>0.5 = el ensemble prefiere el sentimiento en ese par, sin haberlo mirado")
    print(" nunca en el periodo que predice.)")
    print(f"\nGuardado en {METRICS}")

if __name__ == "__main__":
    main()
