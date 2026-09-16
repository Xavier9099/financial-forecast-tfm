# 05 — Diccionario de features (Gold market-only)

Todas causales: valor en *t* calculado sólo con datos ≤ *t*. Verificado en `checks_phase1.py`.

## Del activo
| Feature | Definición | Justificación |
|---|---|---|
| `ret_1d/5d/21d/63d` | retorno simple pasado a h sesiones | momentum y reversión en distintas escalas |
| `vol_21d`, `vol_63d` | desv. típica de `ret_1d` | régimen de riesgo del activo |
| `vol_ratio_21_63` | corto / largo | detecta expansión o contracción de volatilidad |
| `px_vs_sma20/50/200` | precio / media − 1 | tendencia, escala-invariante entre activos |
| `mom_21_63` | `ret_21d − ret_63d` | aceleración del momentum |
| `rsi_14` | RSI de Wilder (EWM recursivo) | sobrecompra/sobreventa; indicador estándar |
| `volchg_5_63` | volumen medio 5d / 63d − 1 | intensidad de actividad anómala |
| `range_21d` | máx(high)/mín(low) − 1 | amplitud realizada |
| `dist_52w_high` | precio / máx 252d − 1 | proximidad a máximos, efecto documentado |

## De contexto
| Feature | Definición |
|---|---|
| `spy_ret_1d/5d/21d/63d`, `spy_vol_21d` | estado del mercado |
| `vix_level`, `vix_chg_5d`, `vix_z_252` | nivel y anomalía de volatilidad implícita |
| `fx_ret_1d/5d/21d`, `fx_vol_21d` | dinámica del USD/MXN |
| `excess_ret_21d/63d` | retorno del activo − retorno de SPY (fuerza relativa) |

**Criterio de admisión**: no se añade ningún indicador sin justificación económica escrita aquí.

## Targets
`target_return_{h}d` = `adj_close[t+h]/adj_close[t] − 1` ·
`target_fx_{h}d` análogo sobre USD/MXN ·
`target_return_mxn_{h}d` = `(1+r_usd)(1+r_fx) − 1` ·
`target_dir_{h}d` = `1` si `r_usd > 0`.
