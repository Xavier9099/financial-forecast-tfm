# 02 — Alcance congelado (MVP)

## Dentro
- Activos: **AAPL, MSFT, JPM**. Benchmark: **SPY**. Contexto: **^VIX**, **USDMXN=X**.
- Horizontes: **5, 21, 63** sesiones.
- Targets: `target_return_{h}d` (USD), `target_fx_{h}d`, `target_return_mxn_{h}d`, `target_dir_{h}d`.
- Modelos: baseline zero-return, LightGBM, XGBoost (+ Ridge si es barato).
- Validación: split temporal + walk-forward con purga del horizonte.
- Experimentos: (1) market-only, (2) market + sentimiento, (3) ensemble dinámico.
- Señales BUY/HOLD/SELL con umbral documentado.
- Demo Streamlit.

## Fuera — POST-TFM
NeuralProphet · TimeGPT · LSTM · Transformers para forecasting · ARIMA masivo ·
optimización de cartera · costes de transacción realistas · intradía · más tickers ·
Azure/Databricks/Delta/MLflow · FastAPI · React · reentrenamiento automático · monitorización.

## Regla de decisión
Ante cualquier propuesta nueva: ¿es indispensable para entregar? Si no → `POST-TFM`
en `12_POST_TFM_ROADMAP.md` y seguir.
