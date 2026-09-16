# 08 — Registro de decisiones

| # | Fecha | Decisión | Motivo | Alternativas descartadas | Impacto |
|---|---|---|---|---|---|
| 1 | 2026-09-12 | Predecir **retorno**, no precio | el precio es no estacionario; un modelo de precio obtiene R² altísimo y engañoso | target = precio futuro | métricas honestas y comparables entre activos |
| 2 | 2026-09-12 | Modalidad **Opción 1** de la guía (Data Scientist) | no requiere aprobación previa de tutores | Opción 3 (propuesta libre) | evita depender de una autorización a pocos días de la entrega |
| 3 | 2026-09-12 | Calendario maestro = sesiones de **SPY** | define sin ambigüedad qué es "un día" en el panel | calendario por activo; calendario natural | alineación reproducible de VIX y FX |
| 4 | 2026-09-12 | `ffill` (límite 5d) para VIX y USD/MXN | sólo mira al pasado | interpolación (usa el futuro), `dropna` | evita look-ahead por construcción |
| 5 | 2026-09-12 | Datos **no versionados** en Git | derechos de uso de la fuente + tamaño | subir los parquet | reproducibilidad vía código + manifest |
| 6 | 2026-09-12 | `adj_close` para todos los retornos | incorpora splits y dividendos | `close` | retorno total, comparable entre activos |
| 7 | 2026-09-12 | Test **point-in-time** automático de look-ahead | prueba directa de causalidad de las features | inspección manual | argumento metodológico fuerte para la defensa |
| 8 | 2026-09-12 | **Purga de `h` sesiones** entre train y test | sin ella, las últimas filas de train tienen target solapado con el test | corte temporal simple | elimina la fuga más común en TFM de series financieras |
| 9 | 2026-09-12 | Backtest financiero **sin solapamiento** (1 obs. cada `h`) | usar predicciones diarias de un target a 63d cuenta el mismo movimiento 63 veces | usar todas las predicciones | evita inflar artificialmente el Sharpe |
| 10 | 2026-09-12 | Umbral de señal = `0.25 × vol_21d × √h` | el listón se adapta al régimen de volatilidad de cada activo | umbral fijo en % | señales comparables entre activos y horizontes |
| 11 | 2026-09-12 | **Hiperparámetros fijos**, sin búsqueda | optimizar sobre el periodo de test es otra forma de fuga; además consume tiempo | GridSearch / Optuna | resultados conservadores y defendibles |
| 12 | 2026-09-12 | Añadir `base_rate` a todas las tablas de `dir_acc` | sin la tasa base, un modelo que siempre predice "sube" parece brillante | reportar `dir_acc` a secas | ver [[09_RESULTS]] H2 |
| 13 | 2026-09-12 | Fuente de noticias: **Alpaca News API** (Benzinga) | 11 años de histórico con timestamp; 5/5 sondas correctas | GDELT (HTTP 429, ritmo inviable), yfinance (sólo días recientes), NewsAPI (1 mes) | Experimento 2 con ventana 2015-2026 |
| 14 | 2026-09-12 | Noticia tras las **16:00 ET** → siguiente sesión | una noticia posterior al cierre no puede informar una predicción hecha al cierre | asignar por fecha natural UTC | regla anti-look-ahead del Experimento 2 |
| 15 | 2026-09-12 | Medias de sentimiento **ponderadas por volumen** | un día con 40 titulares no puede pesar igual que uno con 1 | media simple de medias diarias | agregación representativa de la cobertura real |
| 16 | 2026-09-12 | Días sin noticias → polaridad **0**, no arrastre | rellenar hacia atrás sería mirar al futuro; la ausencia de cobertura es información | `ffill` del sentimiento; eliminar esos días | causalidad preservada; se documenta como limitación |
| 17 | 2026-09-12 | Pool del ensemble: **ElasticNet, Huber, PLS, LightGBM reg., XGBoost** | diversidad real de familias y de patrones de fallo (ver [[09_RESULTS]] H5) | incluir todos los modelos | Ridge sale por redundancia con ElasticNet; `lgbm` sale por estar dominado por `lgbm_reg`; XGBoost se conserva para comprobar que el ensemble lo penaliza |
| 18 | 2026-09-12 | Ensemble calculado sobre `e1_walkforward.parquet`, **sin reentrenar** | las predicciones ya son fuera de muestra y por pliegue | reentrenar modelos dentro del ensemble | ejecución en segundos y auditoría trivial |
| 19 | 2026-09-12 | **TimeGPT** fuera de la comparación principal | paradigma univariante incompatible con un problema multivariante de sección cruzada; en walk-forward exigiría miles de llamadas | incluirlo como un modelo más de la tabla | pasa a anexo/demostración acotada y a líneas futuras |
