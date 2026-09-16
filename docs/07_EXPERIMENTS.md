# 07 — Registro de experimentos

Protocolo común a todos: walk-forward de ventana expansiva, 15 pliegues,
train mínimo 1.260 sesiones, test de 252 sesiones, **purga de `h` sesiones**
entre train y test. Sin búsqueda de hiperparámetros. Semilla 42.

---

## E1 — Market-only ✅

| Campo | Valor |
|---|---|
| Features | 29 (activo, SPY, VIX, USD/MXN, fuerza relativa) — ver [[05_FEATURES]] |
| Targets | `target_return_{5,21,63}d` |
| Tickers | AAPL, MSFT, JPM |
| Train | expansivo desde 2006-01-03 |
| Test | 2011-01-04 → 2026-01-14 |
| Modelos | zero, hist_mean, ridge, elasticnet, huber, pls, lgbm, lgbm_reg, xgb |
| Predicciones | 306.180 fuera de muestra |
| Salida | `outputs/predictions/e1_walkforward.parquet` |

### Resumen predictivo (media entre tickers)

| h | modelo | MAE | RMSE | dir_acc | base | vs base | IC |
|---|---|---|---|---|---|---|---|
| 5 | zero | 0.0270 | 0.0360 | — | 0.576 | — | — |
| 5 | hist_mean | 0.0266 | 0.0358 | 0.576 | 0.576 | 0.000 | −0.029 |
| 5 | lgbm_reg | 0.0272 | 0.0365 | 0.541 | 0.576 | −0.035 | +0.007 |
| 5 | pls | 0.0275 | 0.0368 | 0.538 | 0.576 | −0.038 | **+0.036** |
| 5 | elasticnet | 0.0286 | 0.0381 | 0.530 | 0.576 | −0.046 | +0.033 |
| 21 | zero | 0.0568 | 0.0730 | — | 0.623 | — | — |
| 21 | lgbm_reg | 0.0578 | 0.0745 | 0.549 | 0.623 | −0.074 | −0.011 |
| 21 | pls | 0.0587 | 0.0757 | 0.562 | 0.623 | −0.061 | **+0.060** |
| 21 | ridge | 0.0618 | 0.0798 | 0.541 | 0.623 | −0.081 | +0.056 |
| 63 | zero | 0.1072 | 0.1357 | — | 0.693 | — | — |
| 63 | lgbm_reg | 0.1068 | 0.1350 | 0.603 | 0.693 | −0.090 | +0.022 |
| 63 | pls | 0.1089 | 0.1381 | 0.599 | 0.693 | −0.094 | +0.074 |
| 63 | elasticnet | 0.1134 | 0.1452 | 0.596 | 0.693 | −0.098 | **+0.089** |

### IC de Spearman por ticker (tabla clave)

| ticker | h | elasticnet | huber | pls | ridge | lgbm | lgbm_reg | xgb |
|---|---|---|---|---|---|---|---|---|
| AAPL | 5 | +0.005 | −0.007 | +0.010 | +0.003 | −0.028 | −0.034 | −0.043 |
| AAPL | 21 | +0.038 | +0.029 | +0.051 | +0.042 | −0.000 | −0.014 | −0.014 |
| AAPL | 63 | +0.048 | +0.006 | **+0.081** | +0.045 | +0.039 | +0.049 | +0.059 |
| JPM | 5 | +0.045 | +0.050 | +0.022 | +0.044 | −0.003 | −0.014 | +0.002 |
| JPM | 21 | +0.118 | +0.109 | +0.116 | +0.118 | +0.019 | +0.031 | +0.015 |
| JPM | 63 | +0.237 | **+0.248** | +0.150 | +0.234 | +0.078 | +0.105 | +0.083 |
| MSFT | 5 | +0.049 | +0.048 | **+0.077** | +0.045 | +0.056 | +0.069 | +0.061 |
| MSFT | 21 | +0.009 | +0.000 | +0.013 | +0.007 | −0.087 | −0.051 | −0.078 |
| MSFT | 63 | −0.017 | −0.007 | −0.009 | −0.014 | −0.074 | −0.090 | −0.079 |

### Conclusiones

Ver [[09_RESULTS]] H1 a H6. En una línea: **ningún modelo domina en todos los activos y
horizontes**, lo que fundamenta empíricamente el ensemble dinámico.

---

## E2 — Market + Sentiment ⬜

| Campo | Valor |
|---|---|
| Features | 29 de mercado + 10 de sentimiento — ver [[05_FEATURES]] |
| Fuente | Alpaca News API (Benzinga), 55.939 titulares |
| Ventana | 2015 → 2026 (limitada por la cobertura de noticias) |
| Comparación | E1 y E2 **reentrenados sobre la misma ventana**, nunca E1 completo vs E2 corto |
| Pregunta | ¿el sentimiento aporta capacidad predictiva incremental? |

Resultados: pendiente.

---

## E3 — Ensemble dinámico ⬜

| Campo | Valor |
|---|---|
| Pool | elasticnet, huber, pls, lgbm_reg, xgb ([[08_DECISIONS]] #17) |
| Ponderación | inverse-MAE + acierto direccional, estimada **sólo con pliegues anteriores** |
| Granularidad | por ticker × horizonte |
| Referencias | mejor modelo individual, media simple, zero-return, buy & hold, SPY |

Resultados: pendiente.
