# 03 — Arquitectura técnica

## Flujo
```
Yahoo Finance ──┐
News API ───────┼──▶ BRONZE (parquet crudo por símbolo/fuente + manifest.json)
                │
                └──▶ SILVER (panel limpio, calendario NYSE, ffill causal)
                        │
                        ▶ GOLD  (features causales + targets)  ──▶ modelos
                                                                    │
                                       walk-forward backtesting ◀───┘
                                                │
                                    ensemble dinámico (pesos point-in-time)
                                                │
                              señales BUY/HOLD/SELL ──▶ Streamlit + outputs/
```

## Decisiones de alineación temporal
- Calendario maestro = sesiones de **SPY** (NYSE).
- VIX y USD/MXN reindexados a ese calendario con **forward-fill** (límite 5 días).
  El ffill sólo mira hacia atrás → no introduce look-ahead.
- Todas las features de la fila *t* se calculan con información ≤ cierre de *t*.

## Capas
| Capa | Regla |
|---|---|
| Bronze | Inmutable. Nada se transforma. Se puede reconstruir todo desde aquí. |
| Silver | Limpieza + alineación. Sin features. Sin targets. |
| Gold | Features + targets. Es lo único que consumen los modelos. |

## Arquitectura futura (no se implementa)
Azure Data Lake · Databricks · Spark · Delta Lake · MLflow · PostgreSQL · FastAPI ·
React · scheduling · model serving · monitoring · reentrenamiento automático.
