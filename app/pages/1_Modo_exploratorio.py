"""Modo exploratorio — cualquier activo introducido por el usuario."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import TICKERS  # noqa: E402
from src.exploratory.predict_any import predict  # noqa: E402

st.set_page_config(page_title="Modo exploratorio", layout="wide")
COLORS = {"BUY": "#1a7f37", "SELL": "#c1121f", "HOLD": "#8a8a8a"}

st.title("Modo exploratorio")
st.warning(
    "Los activos del estudio (AAPL, MSFT, JPM) cuentan con **15 pliegues de validación "
    "y 306.180 predicciones fuera de muestra**. Aquí la validación se reduce a "
    "**3 pliegues**. La predicción es orientativa y no equivale en fiabilidad a las del "
    "sistema principal. No constituye asesoramiento financiero.")

c1, c2 = st.columns([3, 1])
symbol = c1.text_input("Símbolo bursátil", value="NVDA",
                       help="Cualquier símbolo de Yahoo Finance: NVDA, KO, TSLA, "
                            "BABA, VOO…").strip().upper()
go = c2.button("Analizar", type="primary", use_container_width=True)

if symbol in TICKERS:
    st.info(f"{symbol} forma parte del estudio principal. Consúltalo en la página "
            f"principal, donde la validación es completa.")

if go and symbol:
    try:
        with st.spinner(f"Descargando {symbol}, construyendo variables y validando…"):
            r = predict(symbol)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"No se pudo analizar {symbol}. Comprueba que el símbolo existe en "
                 f"Yahoo Finance.\n\n{type(e).__name__}: {e}")
        st.stop()

    st.subheader(f"{r['ticker']} · {r['date']:%d/%m/%Y}")
    m = st.columns(3)
    m[0].metric("Precio (USD)", f"${r['price']:,.2f}")
    m[1].metric("Precio (MXN)", f"${r['price'] * r['usdmxn']:,.2f}")
    m[2].metric("Sesiones utilizables", f"{r['n_sessions']:,}")

    tabs = st.tabs([f"{h} sesiones" for h in r["horizons"]])
    for tab, (h, d) in zip(tabs, r["horizons"].items()):
        with tab:
            st.markdown(
                f"<div style='background:{COLORS[d['signal']]};color:white;"
                f"padding:12px 18px;border-radius:8px;font-size:22px;font-weight:600'>"
                f"{d['signal']}</div>", unsafe_allow_html=True)

            c = st.columns(4)
            c[0].metric("Retorno esperado", f"{d['expected_return']:+.2%}")
            c[1].metric("Precio esperado", f"${d['expected_price']:,.2f}",
                        delta=f"{d['expected_price'] - r['price']:+,.2f}")
            c[2].metric("Umbral", f"±{d['threshold']:.2%}")
            c[3].metric("Acuerdo entre modelos", f"{d['agreement']:.0%}")

            ic = d["best_est_ic"]
            if ic > 0.05:
                st.success(f"Capacidad predictiva estimada en este activo: **{ic:+.3f}**. "
                           f"Comparable a la de los activos del estudio.")
            elif ic > 0:
                st.info(f"Capacidad predictiva estimada: **{ic:+.3f}**. Muy débil; "
                        f"la señal apenas se distingue del ruido.")
            else:
                st.error(f"Capacidad predictiva estimada: **{ic:+.3f}**. Negativa: "
                         f"el sistema **no funciona** en este activo. Ignora la señal.")
            st.caption(f"Estimada sobre {d['n_folds']} pliegues de validación. Con tan "
                       f"pocos pliegues, esta cifra es ruidosa y puede estar inflada.")

            w = pd.DataFrame({"peso": d["weights"],
                              "IC estimado": d["est_ic"],
                              "predicción": d["individual"]}).sort_values(
                "peso", ascending=False)
            st.dataframe(w.style.format({"peso": "{:.1%}", "IC estimado": "{:+.3f}",
                                         "predicción": "{:+.2%}"}),
                         use_container_width=True)
            st.caption("Los pesos se asignan según la capacidad predictiva estimada de "
                       "cada modelo en este activo, no según una regla fija.")

with st.expander("Cómo funciona el modo exploratorio"):
    st.markdown("""
1. Descarga el histórico del activo y lo alinea al calendario de sesiones del estudio.
2. Construye **las mismas 29 variables** que el sistema principal, con las mismas funciones.
3. Ejecuta un walk-forward reducido de 3 pliegues, con purga del horizonte, para estimar
   la capacidad predictiva **en ese activo concreto**.
4. Pondera los cinco modelos según esa estimación y entrena la versión final.

La diferencia con el sistema principal no está en el método, sino en la **cantidad de
validación**: 3 pliegues frente a 15. Por eso se muestra siempre la capacidad predictiva
estimada: es el dato que permite juzgar si la señal merece atención.
""")
