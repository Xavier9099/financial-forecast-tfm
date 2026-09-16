"""Demo — Sistema adaptativo multimodal de predicción de rendimientos.

Consume exclusivamente los artefactos generados por el pipeline (outputs/ y data/gold/).
No entrena nada: demuestra que el modelo es utilizable como aplicación, que es el
criterio de "productivización" de la guía del TFM.

    streamlit run app/demo.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GOLD, METRICS, PREDICTIONS, TICKERS  # noqa: E402

st.set_page_config(page_title="TFM — Predicción de rendimientos", layout="wide")

COLORS = {"BUY": "#1a7f37", "SELL": "#c1121f", "HOLD": "#8a8a8a"}
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "JPM": "JPMorgan Chase"}


@st.cache_data
def load():
    d = {}
    d["snap"] = pd.read_csv(METRICS / "latest_snapshot.csv", parse_dates=["date"])
    d["signals"] = pd.read_parquet(PREDICTIONS / "signals.parquet")
    d["weights"] = pd.read_csv(METRICS / "e3_weights.csv")
    try:
        d["gold"] = pd.read_parquet(GOLD / "market_sentiment.parquet")
    except Exception:
        d["gold"] = pd.read_parquet(GOLD / "market_only.parquet")
    return d


try:
    D = load()
except Exception as e:
    st.error(f"Faltan artefactos del pipeline. Ejecuta run_phase1/2/3 y los módulos "
             f"de ensemble antes de la demo.\n\n{e}")
    st.stop()

st.title("Sistema adaptativo multimodal de predicción de rendimientos")
st.caption("TFM · Máster en Big Data, Data Science e IA (UCM) · Xavier Gutiérrez Palma")

with st.sidebar:
    st.header("Parámetros")
    ticker = st.selectbox("Activo", TICKERS, format_func=lambda t: f"{t} — {NAMES[t]}")
    horizon = st.selectbox("Horizonte", [5, 21, 63],
                           format_func=lambda h: f"{h} sesiones ≈ "
                                                 f"{ {5:'1 semana', 21:'1 mes', 63:'3 meses'}[h] }")
    st.divider()
    st.caption("Resultados experimentales con fines académicos. "
               "**No constituyen asesoramiento financiero.**")

row = D["snap"].query("ticker == @ticker and horizon == @horizon")
if row.empty:
    st.warning("No hay predicción disponible para esa combinación.")
    st.stop()
r = row.iloc[0]

sig = r["signal"]
st.markdown(
    f"<div style='background:{COLORS[sig]};color:white;padding:14px 20px;"
    f"border-radius:8px;font-size:26px;font-weight:600'>{sig} · {ticker} · "
    f"{horizon} sesiones</div>", unsafe_allow_html=True)
st.caption(f"Predicción con datos hasta el cierre del {r['date']:%d/%m/%Y}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio actual (USD)", f"${r['adj_close']:,.2f}")
c2.metric("Retorno esperado (USD)", f"{r['expected_return_usd']:+.2%}")
c3.metric("Precio esperado (USD)", f"${r['expected_price_usd']:,.2f}",
          delta=f"{r['expected_price_usd'] - r['adj_close']:+,.2f}")
c4.metric("Umbral de señal", f"±{r['threshold']:.2%}",
          help="0.25 × volatilidad 21d × √horizonte. Se adapta al régimen del activo.")

st.subheader("Perspectiva del inversionista mexicano")
m1, m2, m3 = st.columns(3)
m1.metric("Tipo de cambio USD/MXN", f"${r['usdmxn']:,.2f}")
m2.metric("Retorno esperado (MXN)", f"{r['expected_return_mxn']:+.2%}")
m3.metric("Precio esperado (MXN)",
          f"${r['expected_price_usd'] * r['usdmxn']:,.2f}")
st.caption("La expectativa en pesos asume paseo aleatorio para el tipo de cambio "
           "(Meese y Rogoff, 1983). El análisis ex-post del efecto divisa está en la memoria: "
           "entre el 12.8 % y el 15.7 % de los resultados cambian de signo al convertirse a MXN.")

st.subheader("Confianza y composición del ensemble")
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Acuerdo entre modelos", f"{r['agreement']:.0%}",
              help="Proporción de candidatos que coinciden en signo con el ensemble. "
                   "Se exige ≥ 60 % para emitir BUY o SELL.")
    st.metric("Dispersión de candidatos", f"{r['candidate_dispersion']:.4f}",
              help="Desviación típica entre las predicciones individuales.")
with c2:
    w = D["weights"].query(
        "source == 'e1' and ticker == @ticker and horizon == @horizon "
        "and scheme == 'softmax_ic' and window == 'last3'")
    if not w.empty:
        w = w[w["fold"] == w["fold"].max()][["model", "weight"]]
        w = w.set_index("model").sort_values("weight", ascending=False)
        st.bar_chart(w, horizontal=True)
        st.caption("Pesos del último pliegue, estimados sólo con pliegues anteriores.")

st.subheader("Historial de señales")
h = (D["signals"].query("ticker == @ticker and horizon == @horizon")
     .sort_values("date").tail(500))
if not h.empty:
    st.line_chart(h.set_index("date")[["adj_close"]], height=240)
    dist = h["signal"].value_counts(normalize=True)
    d1, d2, d3 = st.columns(3)
    for col, s in zip((d1, d2, d3), ("BUY", "HOLD", "SELL")):
        col.metric(s, f"{dist.get(s, 0):.0%}")

if "sentiment_mean_7d" in D["gold"].columns:
    st.subheader("Sentimiento reciente de noticias (FinBERT)")
    g = (D["gold"].query("ticker == @ticker").sort_values("date").tail(120))
    s1, s2 = st.columns([2, 1])
    s1.line_chart(g.set_index("date")[["sentiment_mean_7d"]], height=200)
    last = g.iloc[-1]
    s2.metric("Polaridad 7 sesiones", f"{last['sentiment_mean_7d']:+.3f}")
    s2.metric("Titulares (7 sesiones)", f"{int(last['news_volume_7d'])}")

with st.expander("Metodología"):
    st.markdown("""
- **Objetivo**: se predice el **rendimiento futuro**, no el precio. El precio esperado es
  una lectura derivada para facilitar la interpretación.
- **Validación**: walk-forward de ventana expansiva con **purga del horizonte** entre
  entrenamiento y prueba. Ninguna predicción usa datos posteriores a su fecha.
- **Ensemble**: reponderación de cinco modelos (ElasticNet, Huber, PLS, LightGBM
  regularizado, XGBoost) según su IC de Spearman en los **tres pliegues anteriores**.
- **Sentimiento**: FinBERT sobre 55.918 titulares. Las noticias posteriores al cierre
  se asignan a la sesión siguiente.
- **Limitaciones**: ninguna estrategia basada en estas señales supera a *buy & hold*
  en el periodo analizado. El sistema aporta ordenación de expectativas, no rentabilidad
  superior al mercado.
""")
