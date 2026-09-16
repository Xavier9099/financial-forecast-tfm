# 09 — Resultados y hallazgos

> Registro vivo. Cada hallazgo lleva la evidencia que lo sostiene y su lectura de negocio.
> Fuente de los números: `outputs/metrics/e1_predictive.csv`, `e1_financial.csv`.

## Base experimental

- **306.180 predicciones fuera de muestra** (15 pliegues × 3 tickers × 3 horizontes × 9 modelos).
- Periodo de evaluación: **2011-01-04 → 2026-01-14**, ventana expansiva con purga del horizonte.
- Ningún punto de test se usó nunca en entrenamiento; ver [[08_DECISIONS]] #1 y #8.

---

## H1 — Ningún modelo estático bate al modelo nulo en error absoluto

| Horizonte | Mejor MAE relativo (modelo) | Peor |
|---|---|---|
| 5 d | 0.998 — `lgbm_reg` (MSFT) | 1.052 — `lgbm` (AAPL) |
| 21 d | 1.008 — `lgbm_reg` (AAPL) | 1.158 — `elasticnet` (JPM) |
| 63 d | 0.973 — `lgbm_reg` (JPM) | 1.137 — `huber` (AAPL) |

**Lectura.** MAE y RMSE premian el encogimiento hacia cero: cuando la señal es débil,
el predictor que "no se moja" gana. Es una propiedad de la métrica, no una medida de
utilidad de la señal. **Conclusión metodológica: el error cuadrático es la métrica
equivocada para evaluar una señal de inversión.** Demostrado empíricamente, no citado.

---

## H2 — La *directional accuracy* es engañosa sin su tasa base

`hist_mean` alcanza **0.693** de acierto direccional a 63 días. No tiene habilidad
predictiva: siempre predice positivo, y el 69,3 % de los retornos a 63 días del periodo
fueron positivos. Su acierto **es** la tasa base.

| Horizonte | Tasa base | Mejor modelo ML | Diferencia |
|---|---|---|---|
| 5 d | 0.576 | 0.541 (`lgbm_reg`) | **−0.035** |
| 21 d | 0.623 | 0.562 (`pls`) | **−0.061** |
| 63 d | 0.693 | 0.604 (`huber`) | **−0.089** |

**Lectura.** Todos los modelos aciertan la dirección **peor** que estar siempre largo.
Compatible con un IC positivo: el modelo ordena bien la magnitud relativa aunque falle
el signo, que es lo que explota una estrategia con umbral. Reportar ambas métricas juntas
es obligatorio; reportar sólo `dir_acc` sería engañoso.

---

## H3 — La señal existe, es débil, es lineal y crece con el horizonte

IC de Spearman medio entre tickers:

| Modelo | 5 d | 21 d | 63 d |
|---|---|---|---|
| PLS | +0.036 | +0.060 | +0.074 |
| ElasticNet | +0.033 | +0.055 | **+0.089** |
| Ridge | +0.031 | +0.056 | +0.088 |
| Huber | +0.030 | +0.046 | +0.083 |
| LightGBM | +0.009 | −0.023 | +0.014 |
| XGBoost | +0.007 | −0.026 | +0.021 |

**Lectura.** Los modelos lineales regularizados extraen señal; los de árboles ajustan
ruido. Coherente con la literatura de momentum a horizontes largos. Un IC de 0.09 es
modesto pero está en el rango aprovechable con una cartera diversificada.

---

## H4 — Menos capacidad, mejor resultado (evidencia directa)

`lgbm` y `lgbm_reg` son **el mismo algoritmo**; sólo cambia la capacidad
(31→7 hojas, `lr` 0.03→0.01, `reg_lambda` 1→10).

| | MAE rel. medio | IC medio 63 d |
|---|---|---|
| `lgbm` | 1.082 | +0.014 |
| `lgbm_reg` | **1.011** | **+0.022** |

**Lectura.** En un dominio con ratio señal/ruido bajo, la capacidad adicional se
consume modelando ruido. Justifica no escalar a arquitecturas profundas
(ver [[12_POST_TFM_ROADMAP]]) con evidencia propia, no con una opinión.

---

## H5 — Ningún modelo domina: heterogeneidad por activo y horizonte ⭐

IC de Spearman, celdas seleccionadas:

| | AAPL 63d | JPM 63d | MSFT 63d | MSFT 5d |
|---|---|---|---|---|
| ElasticNet | +0.048 | **+0.237** | −0.017 | +0.049 |
| Huber | +0.006 | **+0.248** | −0.007 | +0.048 |
| PLS | +0.081 | +0.150 | −0.009 | +0.077 |
| LightGBM reg. | +0.049 | +0.105 | −0.090 | **+0.069** |

**Lectura — este es el hallazgo central del TFM.** JPM a 63 días tiene señal fuerte y
lineal; MSFT a 63 días no tiene señal en ningún modelo; en MSFT a 5 días gana un modelo
de árboles, la familia que peor se comporta en el resto. **La hipótesis de partida deja
de ser una suposición razonable y pasa a ser un hallazgo medido**: el ensemble adaptativo
es la respuesta lógica a un patrón que los datos imponen.

⚠️ **Cautela estadística obligatoria.** Con targets solapados a 63 días, la muestra
efectiva no son 3.780 observaciones sino ~60 independientes. El IC de 0.248 de JPM tiene
un intervalo de confianza ancho. Reportar como señal prometedora, **no** como resultado
robusto.

---

## H6 — Buy & hold domina en Sharpe

Sharpe anualizado de buy & hold: 0.74 a 1.14 según ticker y horizonte. Ninguna estrategia
basada en modelo lo supera de forma consistente.

**Lectura.** Esperable en quince años de mercado alcista. Va a conclusiones sin adornos:
el sistema aporta información sobre la ordenación esperada de retornos, no una estrategia
que bata al mercado. Honestidad > espectacularidad.

---

## Pendiente

- [ ] E2 — Market-only vs Market+Sentiment sobre ventana común
- [ ] E3 — Ensemble dinámico
- [ ] Descomposición USD/MXN
- [ ] Métricas financieras de las señales BUY/HOLD/SELL
