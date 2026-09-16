# 00 — Contexto del proyecto (resumen vivo)

> Punto de entrada del vault. Si sólo se lee un archivo, que sea este.

## Qué es
Sistema experimental de apoyo a decisiones financieras que predice **rendimientos futuros**
(no precios) a 5, 21 y 63 sesiones para AAPL, MSFT y JPM, combinando features de mercado,
VIX, USD/MXN y sentimiento de noticias (FinBERT), con un ensemble dinámico entre varios
modelos, validado mediante walk-forward backtesting con purga del horizonte.

## Hipótesis central
Un ensemble adaptativo que repondera modelos según su desempeño reciente puede comportarse
mejor que mantener un único modelo estático, porque el comportamiento financiero cambia
según activo, horizonte y régimen de mercado.

**Estado de la hipótesis: respaldada empíricamente en su premisa.** [[09_RESULTS]] H5
demuestra que ningún modelo domina en todos los pares activo × horizonte.

## Aportación
1. Panel multimodal construido desde cero (precio + macro + divisa + NLP), no descargado.
2. Ensemble dinámico con pesos point-in-time.
3. Perspectiva del inversionista mexicano: retorno descompuesto en activo × divisa.
4. Protocolo de validación con purga y **test automático de look-ahead**.
5. Productivización: demo Streamlit sobre el pipeline real.

## Cifras del proyecto
| | |
|---|---|
| Panel de mercado | 15.612 filas × 29 features, 2006-01-03 → 2026-09-10 |
| Titulares | 55.939 (Alpaca/Benzinga), 2015 → 2026 |
| Predicciones fuera de muestra | 306.180 |
| Pliegues walk-forward | 15 por ticker × horizonte |
| Validaciones automáticas de integridad | 6, todas en verde |

## Estado
| Bloque | Estado |
|---|---|
| 1 — Data foundation + validación de fugas | ✅ |
| 2 — Modelos + walk-forward + E1 | ✅ |
| 3 — FinBERT → sentimiento → Gold definitivo → E2 | 🟡 |
| 4 — Ensemble dinámico + USD/MXN + señales | ⬜ |
| 5 — Streamlit | ⬜ |
| 6 — Congelado de resultados | ⬜ |
| 7 — Memoria + README + anexos | ⬜ |
| 8 — Vídeo | ⬜ |
| 9 — QA y entrega | ⬜ |

## Mapa del vault
[[01_TFM_REQUIREMENTS]] requisitos oficiales · [[02_SCOPE]] alcance congelado ·
[[03_ARCHITECTURE]] arquitectura · [[04_DATA_SOURCES]] fuentes y derechos ·
[[05_FEATURES]] diccionario · [[06_MODELS]] modelos · [[07_EXPERIMENTS]] experimentos ·
[[08_DECISIONS]] decisiones · [[09_RESULTS]] hallazgos · [[10_TFM_DRAFT]] borrador ·
[[11_VIDEO_SCRIPT]] guion · [[12_POST_TFM_ROADMAP]] líneas futuras

## Regla de cierre
El congelado de resultados se hace **una sola vez**. Después no se toca el pipeline: si
los números cambian tras empezar la memoria, hay que reescribirla.
