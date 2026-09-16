"""INFERENCIA EN VIVO — Predicción con los datos más recientes disponibles.

Distingue dos cosas que no deben confundirse:

  · VALIDACIÓN  ventana congelada (2006 → 2026-09-10), 15 pliegues, 306.180
                predicciones fuera de muestra. Es lo que sostiene la memoria.
  · INFERENCIA  este módulo: descarga el cierre más reciente, entrena con toda la
                historia utilizable y predice la sesión actual. No genera métricas
                nuevas: aplica el sistema ya validado.

Es la separación estándar en producción. Extender la ventana de validación
obligaría a recalcular todos los resultados; extender la de inferencia, no.

Los pesos del ensemble se toman del último pliegue del estudio (esquema por
capacidad predictiva reciente, ventana corta), estimados sin ver el futuro.

Salida: outputs/metrics/latest_snapshot.csv  (la consume la demo)

Uso:  python -m src.exploratory.live_predict
"""
from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from src.config import (BENCHMARK, FX, FX_FALLBACK, HORIZONS, METRICS, START,
                        TICKERS, VIX, WARMUP_DAYS, ensure_dirs)
from src.ensemble.signals import K_THRESHOLD, MIN_AGREEMENT
from src.features.build_gold import (FEATURE_COLS_EXCLUDE, asset_features,
                                     context_features)
from src.models.train_eval import make_models

POOL = ["elasticnet", "huber", "pls", "lgbm_reg", "xgb"]
RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
          "Adj Close": "adj_close", "Volume": "volume"}

def fetch(symbol: str) -> pd.DataFrame:
    """Descarga hasta el último cierre disponible (sin tope congelado)."""
    end = (date.today() + timedelta(days=1)).isoformat()
    df = yf.download(symbol, start=START, end=end, auto_adjust=False,
                     progress=False, actions=False)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        df.columns = lvl0 if set(RENAME).intersection(lvl0) else df.columns.get_level_values(1)
    if df.empty and symbol == FX:
        return fetch(FX_FALLBACK)
    df = df.rename(columns=RENAME).reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    if "adj_close" not in df:
        df["adj_close"] = df["close"]
    if "volume" not in df:
        df["volume"] = np.nan
    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    return df[keep].sort_values("date").drop_duplicates("date").reset_index(drop=True)

def live_context() -> pd.DataFrame:
    spy, vix, fx = fetch(BENCHMARK), fetch(VIX), fetch(FX)
    cal = pd.DatetimeIndex(spy["date"])
    ctx = pd.DataFrame({
        "date": cal,
        "spy_adj_close": spy.set_index("date")["adj_close"].reindex(cal).to_numpy(),
        "vix_close": vix.set_index("date")["close"].reindex(cal).ffill(limit=5).to_numpy(),
        "usdmxn": fx.set_index("date")["close"].reindex(cal).ffill(limit=5).to_numpy(),
    })
    return ctx.dropna().reset_index(drop=True)

def study_weights() -> dict:
    """Pesos del último pliegue del estudio, por ticker y horizonte."""
    try:
        w = pd.read_csv(METRICS / "e3_weights.csv")
    except FileNotFoundError:
        return {}
    w = w[(w["source"] == "e1") & (w["scheme"] == "softmax_ic") & (w["window"] == "last3")]
    if w.empty:
        return {}
    w = w[w["fold"] == w["fold"].max()]
    return {(r.ticker, int(r.horizon)): (r.model, float(r.weight))
            for r in w.itertuples()}

def main() -> None:
    ensure_dirs()
    print("Descargando contexto de mercado más reciente...")
    ctx = live_context()
    ctxf = pd.concat([ctx, context_features(ctx)], axis=1)
    print(f"  calendario hasta {ctx['date'].max():%Y-%m-%d}")

    wmap = study_weights()
    rows = []

    for ticker in TICKERS:
        raw = fetch(ticker)
        raw = raw[raw["date"].isin(set(ctx["date"]))].reset_index(drop=True)
        g = raw[["date", "close", "adj_close", "volume", "high", "low"]]
        d = pd.concat([g, asset_features(g)], axis=1).merge(ctxf, on="date", how="inner")
        d["excess_ret_21d"] = d["ret_21d"] - d["spy_ret_21d"]
        d["excess_ret_63d"] = d["ret_63d"] - d["spy_ret_63d"]
        d = d.iloc[WARMUP_DAYS:].reset_index(drop=True)

        feats = [c for c in d.columns
                 if c not in FEATURE_COLS_EXCLUDE and not c.startswith("target_")]
        X_all = d[feats].to_numpy(dtype=float)
        p = d["adj_close"].astype(float)
        price, fxr = float(p.iloc[-1]), float(d["usdmxn"].iloc[-1])
        last_date = d["date"].iloc[-1]

        for h in HORIZONS:
            y = (p.shift(-h) / p - 1.0).to_numpy(dtype=float)
            ok = np.isfinite(y)
            individual = {}
            for name, model in make_models().items():
                if name not in POOL:
                    continue
                model.fit(X_all[ok], y[ok])
                individual[name] = float(np.asarray(model.predict(X_all[-1:])).ravel()[0])

            sub = {m: wt for (t, hh), (m, wt) in wmap.items()
                   if t == ticker and hh == h}
            w = ({m: sub.get(m, 0.0) for m in POOL} if sub
                 else {m: 1 / len(POOL) for m in POOL})
            tot = sum(w.values()) or 1.0
            w = {m: v / tot for m, v in w.items()}

            pred = sum(w[m] * individual[m] for m in POOL)
            vol = float(d["vol_21d"].iloc[-1])
            thr = K_THRESHOLD * vol * np.sqrt(h)
            agree = float(np.mean([np.sign(individual[m]) == np.sign(pred) for m in POOL]))
            sig = ("BUY" if (agree >= MIN_AGREEMENT and pred > thr)
                   else "SELL" if (agree >= MIN_AGREEMENT and pred < -thr) else "HOLD")

            rows.append({
                "date": last_date, "ticker": ticker, "horizon": h,
                "adj_close": price, "expected_return_usd": pred,
                "expected_price_usd": price * (1 + pred),
                "expected_return_mxn": pred, "usdmxn": fxr, "threshold": thr,
                "agreement": agree,
                "candidate_dispersion": float(np.std(list(individual.values()))),
                "signal": sig,
            })
        print(f"  {ticker} listo · cierre {last_date:%Y-%m-%d} · ${price:,.2f}")

    snap = pd.DataFrame(rows)
    snap.to_csv(METRICS / "latest_snapshot.csv", index=False)

    print(f"\n{'ticker':<6}{'h':>5}{'señal':>8}{'ret.esp.':>11}{'precio esp.':>14}"
          f"{'umbral':>9}{'acuerdo':>9}")
    for r in snap.itertuples():
        print(f"{r.ticker:<6}{r.horizon:>5}{r.signal:>8}{r.expected_return_usd:>+10.2%}"
              f" ${r.expected_price_usd:>12,.2f}{r.threshold:>8.2%}{r.agreement:>8.0%}")
    print(f"\n✅ latest_snapshot.csv actualizado con el cierre de "
          f"{snap['date'].max():%d/%m/%Y}")
    print("   La ventana de VALIDACIÓN sigue congelada; esto es sólo inferencia.")

if __name__ == "__main__":
    main()
