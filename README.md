# Sistema adaptativo multimodal para predicción de rendimientos bursátiles

**TFM — Máster en Big Data, Data Science e Inteligencia Artificial (UCM)**
Autor: Xavier Gutiérrez Palma

Sistema experimental de apoyo a la decisión que combina series temporales financieras,
contexto de mercado (SPY, VIX), divisa (USD/MXN) y sentimiento de noticias (FinBERT)
para predecir **rendimientos futuros** a 5, 21 y 63 sesiones sobre AAPL, MSFT y JPM,
mediante un **ensemble dinámico** validado con *walk-forward backtesting*.

> Los resultados y las señales BUY/HOLD/SELL tienen finalidad **exclusivamente académica
> y experimental**. No constituyen asesoramiento financiero.

## Hipótesis

Dado que el comportamiento financiero cambia según activo, horizonte y régimen de mercado,
un ensemble adaptativo que combine información de mercado y sentimiento puede comportarse
mejor que mantener un único modelo estático.

## Reproducción rápida

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_phase1.py          # Bronze -> Silver -> Gold -> validación de fugas temporales
```

## Arquitectura de datos

| Capa | Contenido | Formato |
|---|---|---|
| **Bronze** | Descargas crudas por símbolo, sin transformar, con metadatos de ingesta | `data/bronze/*.parquet` |
| **Silver** | Panel de precios limpio + contexto de mercado alineado al calendario NYSE | `data/silver/*.parquet` |
| **Gold** | Tabla de features causales + targets de retorno futuro, lista para modelizar | `data/gold/*.parquet` |

Los datos no se versionan (derechos de uso de la fuente); el pipeline los regenera
íntegramente y de forma determinista a partir de `src/config.py`.

## Estructura

```
src/ingestion/   descarga de fuentes externas          app/       demo Streamlit
src/features/    Silver, Gold y feature engineering    docs/      vault Obsidian (memoria)
src/sentiment/   noticias + FinBERT                    outputs/   figuras, métricas, predicciones
src/models/      baseline, LightGBM, XGBoost           notebooks/ EDA y anexos
src/backtesting/ walk-forward                          
src/ensemble/    pesos dinámicos                       
src/validation/  tests de integridad temporal          
```

## Garantías metodológicas

- Sin `random_split`: partición temporal + walk-forward con *purga* del horizonte.
- Toda feature en la fecha *t* usa sólo información disponible al cierre de *t*, verificado
  por un test *point-in-time* que recalcula las features sobre la serie truncada
  (`src/validation/checks_phase1.py`).
- Los pesos del ensemble se estiman únicamente con observaciones anteriores a la fecha predicha.

## Fuentes de datos

| Serie | Símbolo | Fuente |
|---|---|---|
| Acciones | AAPL, MSFT, JPM | Yahoo Finance (`yfinance`) |
| Benchmark | SPY | Yahoo Finance |
| Volatilidad implícita | ^VIX | Yahoo Finance (CBOE) |
| Divisa | USDMXN=X | Yahoo Finance |
| Noticias | por ticker | ver `docs/04_DATA_SOURCES.md` |
