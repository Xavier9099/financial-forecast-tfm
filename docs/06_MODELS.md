# 06 — Modelos

| Modelo | Rol | Estado |
|---|---|---|
| Zero-return (naïve) | baseline obligatorio: predice retorno 0 | ⬜ |
| Media histórica móvil | segundo baseline trivial | ⬜ |
| Ridge | lineal regularizado, referencia interpretable | ⬜ |
| LightGBM | no lineal principal | ⬜ |
| XGBoost | no lineal alternativo para el ensemble | ⬜ |
| Ensemble dinámico | reponderación por desempeño reciente | ⬜ |

Un modelo por `ticker × horizonte`. Hiperparámetros conservadores y fijos
(sin búsqueda costosa): la profundidad y el `n_estimators` se limitan por el
bajo ratio señal/ruido de los retornos.

**Fuera de alcance (POST-TFM)**: NeuralProphet, TimeGPT, LSTM, Transformers, ARIMA masivo.
