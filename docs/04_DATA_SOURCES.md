# 04 — Fuentes de datos

| Serie | Símbolo | Fuente | Frecuencia | Notas |
|---|---|---|---|---|
| Acciones | AAPL, MSFT, JPM | Yahoo Finance (`yfinance`) | diaria | se usa `Adj Close` (splits y dividendos) |
| Benchmark | SPY | Yahoo Finance | diaria | define el calendario maestro |
| Volatilidad implícita | ^VIX | Yahoo Finance (CBOE) | diaria | nivel, cambio y z-score 252d |
| Divisa | USDMXN=X | Yahoo Finance | diaria | cotiza más días que el NYSE → se reindexa |
| Noticias | por ticker | **por confirmar (Fase 3)** | intradía | ver spike de viabilidad |

## Derechos de uso
Datos de Yahoo Finance obtenidos mediante la librería abierta `yfinance` para **uso académico
y no comercial**. No se redistribuyen los datos brutos en el repositorio: se versiona el
código de ingesta, que los regenera de forma determinista. `data/bronze/manifest.json`
documenta fechas, filas y momento de descarga de cada serie.

## Ventana
`2005-01-01` → `2026-09-11` (congelada en `src/config.py`). Cubre la crisis de 2008,
el COVID-19 de 2020 y el ciclo de tipos posterior: tres regímenes distintos, útil para
defender la hipótesis de adaptabilidad.
