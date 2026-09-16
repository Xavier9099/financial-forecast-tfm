"""FASE 4b — Señales BUY/HOLD/SELL, perspectiva MXN y métricas financieras.

SEÑALES
  Umbral adaptativo al régimen:  thr = K · vol_21d · √h   (K = 0.25)
  Un retorno esperado del 1 % no significa lo mismo en un activo con volatilidad
  anualizada del 15 % que en uno del 45 %. El umbral escala con la volatilidad
  realizada a 21 sesiones, conocida en t.

  Confianza = acuerdo entre candidatos del ensemble (proporción que coincide en signo).
  BUY  : retorno esperado > +thr  y acuerdo ≥ 0.60
  SELL : retorno esperado < −thr  y acuerdo ≥ 0.60
  HOLD : resto

PERSPECTIVA MXN
  Expectativa ex-ante: se asume paseo aleatorio para el tipo de cambio (variación
  esperada 0 %). Es el estándar en la literatura desde Meese y Rogoff (1983): ningún
  modelo bate sistemáticamente al paseo aleatorio en predicción cambiaria a estos
  horizontes. Por tanto, ex-ante, Retorno_MXN esperado = Retorno_USD esperado.

  Análisis ex-post: se descompone el retorno realizado
      (1 + r_MXN) = (1 + r_USD) · (1 + r_FX)
  para cuantificar cuánto aportó la divisa y en cuántos casos cambió el signo del
  resultado para un inversionista mexicano. Ahí está el hallazgo, no en la predicción.

Salidas:
  outputs/predictions/signals.parquet
  outputs/metrics/e4_signals.csv
  outputs/metrics/e4_financial.csv
  outputs/metrics/latest_snapshot.csv   <- la consume la demo de Streamlit

Uso:  python -m src.ensemble.signals
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import (BENCHMARK, GOLD, HORIZONS, METRICS, PREDICTIONS, SILVER,
                        TICKERS, ensure_dirs)
from src.models.metrics import financial_metrics

K_THRESHOLD = 0.25
MIN_AGREEMENT = 0.60
PRIMARY = "ens_softmax_ic_last3"     # mejor esquema ponderado de E3a (08_DECISIONS #20)
POOL = ["elasticnet", "huber", "pls", "lgbm_reg", "xgb"]

def build_signals() -> pd.DataFrame:
    gold = pd.read_parquet(GOLD / "market_only.parquet")
    ens = pd.read_parquet(PREDICTIONS / "e3_ensemble.parquet")
    ind = pd.read_parquet(PREDICTIONS / "e1_walkforward.parquet")

    ens = ens[(ens["source"] == "e1") & (ens["model"] == PRIMARY)].copy()

    # Acuerdo entre candidatos: proporción que coincide con el signo del ensemble
    ind = ind[ind["model"].isin(POOL)]
    piv = ind.pivot_table(index=["date", "ticker", "horizon"], columns="model",
                          values="y_pred")
    sgn = np.sign(piv.to_numpy())
    disp = piv.std(axis=1).rename("candidate_dispersion")
    piv = piv.assign(candidate_dispersion=disp).reset_index()

    df = ens.merge(piv, on=["date", "ticker", "horizon"], how="left")
    cand = df[POOL].to_numpy()
    agree = (np.sign(cand) == np.sign(df["y_pred"].to_numpy())[:, None]).mean(axis=1)
    df["agreement"] = agree

    # Contexto de mercado en la fecha de la predicción
    ctx = gold[["date", "ticker", "adj_close", "vol_21d", "usdmxn"]]
    df = df.merge(ctx, on=["date", "ticker"], how="left")

    # Umbral adaptativo y señal
    df["threshold"] = K_THRESHOLD * df["vol_21d"] * np.sqrt(df["horizon"])
    df["signal"] = "HOLD"
    df.loc[(df["y_pred"] > df["threshold"]) & (df["agreement"] >= MIN_AGREEMENT),
           "signal"] = "BUY"
    df.loc[(df["y_pred"] < -df["threshold"]) & (df["agreement"] >= MIN_AGREEMENT),
           "signal"] = "SELL"

    # Interpretación en precio y divisa
    df["expected_return_usd"] = df["y_pred"]
    df["expected_price_usd"] = df["adj_close"] * (1 + df["y_pred"])
    df["expected_return_mxn"] = df["y_pred"]          # paseo aleatorio en FX (ex-ante)

    # Realizado (ex-post) para el análisis de divisa
    tgt = gold[["date", "ticker"] + [f"target_fx_{h}d" for h in HORIZONS]]
    df = df.merge(tgt, on=["date", "ticker"], how="left")
    df["realized_fx"] = [r[f"target_fx_{int(h)}d"] for h, (_, r) in
                         zip(df["horizon"], df.iterrows())]
    df["realized_return_usd"] = df["y_true"]
    df["realized_return_mxn"] = (1 + df["y_true"]) * (1 + df["realized_fx"]) - 1
    return df

def report_signals(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("DISTRIBUCIÓN DE SEÑALES")
    print("=" * 70)
    dist = (df.groupby(["ticker", "horizon"])["signal"]
            .value_counts(normalize=True).unstack().fillna(0).round(3))
    print(dist.to_string())

    print("\n" + "=" * 70)
    print("CALIDAD DE LA SEÑAL (retorno realizado medio, USD)")
    print("=" * 70)
    q = (df.groupby(["horizon", "signal"])["realized_return_usd"]
         .agg(n="count", media="mean", acierto=lambda s: float((s > 0).mean()))
         .round(4))
    print(q.to_string())
    print("\nLectura: BUY debe tener media y acierto superiores a HOLD y SELL.")

def report_fx(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PERSPECTIVA DEL INVERSIONISTA MEXICANO (ex-post)")
    print("=" * 70)
    rows = []
    for h in HORIZONS:
        s = df[df["horizon"] == h]
        flip = ((np.sign(s["realized_return_usd"]) != np.sign(s["realized_return_mxn"]))
                .mean())
        rows.append({
            "horizonte": h,
            "ret_USD_medio": s["realized_return_usd"].mean(),
            "ret_MXN_medio": s["realized_return_mxn"].mean(),
            "aporte_divisa": s["realized_return_mxn"].mean() - s["realized_return_usd"].mean(),
            "vol_USD": s["realized_return_usd"].std(),
            "vol_MXN": s["realized_return_mxn"].std(),
            "cambia_el_signo": flip,
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n`cambia_el_signo` = proporción de casos en que el resultado en pesos tuvo")
    print("signo distinto al resultado en dólares. Es la magnitud del riesgo cambiario")
    print("para un inversionista mexicano que no cubre la divisa.")

def report_financial(df: pd.DataFrame) -> pd.DataFrame:
    prices = pd.read_parquet(SILVER / "prices.parquet")
    spy = (prices[prices["ticker"] == BENCHMARK]
           .set_index("date")["adj_close"].sort_index())

    rec = []
    for (t, h), g in df.groupby(["ticker", "horizon"]):
        g = g.sort_values("date").iloc[::int(h)]          # sin solapamiento
        pos = np.where(g["signal"] == "BUY", 1.0,
                       np.where(g["signal"] == "SELL", -1.0, 0.0))

        for label, rets, p in [
            ("señales_USD", g["realized_return_usd"].to_numpy(), pos),
            ("señales_MXN", g["realized_return_mxn"].to_numpy(), pos),
            ("solo_largos", g["realized_return_usd"].to_numpy(), np.maximum(pos, 0)),
            ("buy_and_hold", g["realized_return_usd"].to_numpy(),
             np.ones(len(g))),
        ]:
            m = financial_metrics(rets, p, int(h))
            if m:
                rec.append({"ticker": t, "horizon": h, "estrategia": label, **m})

        # Benchmark SPY sobre las mismas fechas
        idx = spy.index.searchsorted(g["date"].to_numpy())
        idx2 = np.minimum(idx + int(h), len(spy) - 1)
        spy_ret = spy.to_numpy()[idx2] / spy.to_numpy()[np.minimum(idx, len(spy) - 1)] - 1
        m = financial_metrics(spy_ret, np.ones(len(spy_ret)), int(h))
        if m:
            rec.append({"ticker": t, "horizon": h, "estrategia": "SPY", **m})

    fin = pd.DataFrame(rec)
    print("\n" + "=" * 70)
    print("SHARPE ANUALIZADO — SEÑALES vs REFERENCIAS")
    print("=" * 70)
    print(fin.pivot_table(index=["ticker", "horizon"], columns="estrategia",
                          values="sharpe").round(2).to_string())
    print("\nRETORNO ACUMULADO")
    print(fin.pivot_table(index=["ticker", "horizon"], columns="estrategia",
                          values="cum_return").round(3).to_string())
    print("\nEXPOSICIÓN (proporción del tiempo con posición abierta)")
    print(fin[fin["estrategia"] == "señales_USD"]
          .pivot_table(index="ticker", columns="horizon", values="exposure").round(2).to_string())
    return fin

def main() -> None:
    ensure_dirs()
    df = build_signals()
    df.to_parquet(PREDICTIONS / "signals.parquet", index=False)

    report_signals(df)
    report_fx(df)
    fin = report_financial(df)

    (df.groupby(["ticker", "horizon", "signal"]).size().rename("n").reset_index()
     .to_csv(METRICS / "e4_signals.csv", index=False))
    fin.to_csv(METRICS / "e4_financial.csv", index=False)

    # Última señal disponible por ticker × horizonte -> demo Streamlit
    snap = (df.sort_values("date").groupby(["ticker", "horizon"]).tail(1)
            [["date", "ticker", "horizon", "adj_close", "expected_return_usd",
              "expected_price_usd", "expected_return_mxn", "usdmxn", "threshold",
              "agreement", "candidate_dispersion", "signal"]])
    snap.to_csv(METRICS / "latest_snapshot.csv", index=False)
    print(f"\n Señales, divisa y métricas financieras guardadas en {METRICS}")

if __name__ == "__main__":
    main()
