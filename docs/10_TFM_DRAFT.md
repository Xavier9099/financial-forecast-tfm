# Sistema adaptativo multimodal para la predicción de rendimientos bursátiles mediante ensemble dinámico y análisis de sentimiento

**Trabajo Fin de Máster** · Máster en Big Data, Data Science e Inteligencia Artificial · Universidad Complutense de Madrid
**Autor:** Xavier Gutiérrez Palma · **Tutores:** Carlos Ortega y Santiago Mota

> *Nota de trabajo, eliminar antes de entregar:* verificar cada cifra contra `outputs/metrics/`
> tras la ejecución final. Formato requerido: Arial o Verdana 11, máximo 20 caras sin contar
> portada, índice ni anexos.

---

## 1. El problema y por qué importa

Un inversionista mexicano que quiere colocar ahorro en acciones estadounidenses se enfrenta a tres decisiones simultáneas: qué activo, con qué horizonte y asumiendo qué riesgo cambiario. La industria ofrece dos extremos poco satisfactorios. En uno están los sistemas que prometen predecir el mercado, promesa que la evidencia académica desmiente de forma consistente. En el otro, el consejo de no hacer nada y comprar un índice, que es razonable pero no aporta información sobre decisiones concretas.

Este trabajo explora el espacio intermedio. No pretende predecir el mercado; pretende **medir con rigor cuánta información aprovechable existe** en datos públicos, y construir un sistema que la utilice de forma adaptativa.

### 1.1 Hipótesis

> Dado que el comportamiento financiero cambia según el activo, el horizonte y el régimen de mercado, un ensemble adaptativo que reponderá varios modelos según su desempeño reciente puede comportarse mejor que mantener un único modelo estático.

Obsérvese que la hipótesis **no** afirma que sea posible batir al mercado. Afirma algo más modesto y más comprobable: que la adaptación es preferible a la rigidez.

### 1.2 Qué aporta este trabajo

1. Un conjunto de datos **construido**, no descargado: precios, volatilidad implícita, tipo de cambio y 55.918 titulares procesados con un modelo de lenguaje financiero, alineados temporalmente sin fugas de información.
2. Un protocolo de validación con **purga del horizonte** y una prueba automática de causalidad que verifica, sobre los propios datos, que ninguna variable usa información futura.
3. Un **ensemble dinámico** cuyos pesos se estiman exclusivamente con observaciones anteriores a la fecha que se predice.
4. La **descomposición del rendimiento en activo y divisa** desde la perspectiva de un inversionista que consume en pesos.
5. Una **aplicación funcional** que consume el sistema y devuelve predicciones y señales.

### 1.3 Lo que el lector encontrará

Adelanto la conclusión para que el resto se lea en contexto: **el sistema contiene información predictiva real pero pequeña, y esa información no se traduce en una estrategia que supere a comprar y mantener.** Ambas afirmaciones están sostenidas por 306.180 predicciones fuera de muestra. Un trabajo que concluyera lo contrario debería levantar sospechas de error metodológico.

---

## 2. Datos

### 2.1 Universo y fuentes

| Serie | Símbolo | Fuente | Uso |
|---|---|---|---|
| Acciones | AAPL, MSFT, JPM | Yahoo Finance | activos objeto de predicción |
| Mercado | SPY | Yahoo Finance | referencia y calendario maestro |
| Volatilidad implícita | ^VIX | CBOE vía Yahoo | contexto de riesgo |
| Divisa | USDMXN=X | Yahoo Finance | perspectiva del inversionista mexicano |
| Noticias | 55.918 titulares | Alpaca News API (Benzinga) | sentimiento |

Se eligieron tres empresas de sectores distintos —tecnología de consumo, software empresarial y banca— para que la comparación entre activos no estuviera dominada por un único sector. El periodo, de enero de 2006 a septiembre de 2026, contiene tres regímenes claramente diferenciados: la crisis financiera de 2008, la caída y recuperación de 2020 y el ciclo de tipos posterior. Esa diversidad de regímenes es precisamente lo que permite contrastar la hipótesis.

**Volumen final:** 15.612 filas con 29 variables para el análisis de mercado; 8.817 filas con 39 variables para el análisis con sentimiento.

> **[FIGURA 1]** `fig01_panel_contexto.png` — Precios normalizados, VIX y USD/MXN, 2006-2026.

### 2.2 Derechos de uso

Los datos de mercado se obtienen mediante la librería abierta `yfinance` para uso académico y no comercial. Los titulares proceden de la API de Alpaca bajo cuenta gratuita. **No se redistribuyen los datos brutos**: el repositorio contiene el código de ingesta, que los regenera de forma determinista, y un manifiesto con fechas, número de filas y momento de descarga de cada serie.

### 2.3 Arquitectura Bronze / Silver / Gold

- **Bronze**: descargas crudas, una por símbolo, sin transformación alguna. Capa inmutable.
- **Silver**: limpieza y alineación temporal. El calendario maestro son las sesiones de SPY; VIX y USD/MXN se reindexan a él mediante arrastre hacia adelante limitado a cinco días. El arrastre solo mira hacia atrás, de modo que es causal por construcción. La interpolación, alternativa natural, habría usado valores futuros e introducido una fuga silenciosa.
- **Gold**: variables explicativas y objetivos. Es la única capa que consumen los modelos.

---

## 3. Metodología

Esta sección concentra las decisiones que hacen creíbles o increíbles los resultados. Es deliberadamente la más detallada.

### 3.1 Predecir rendimiento, no precio

El precio de una acción no es estacionario: crece a lo largo del tiempo. Un modelo que prediga el precio de mañana a partir del de hoy obtendrá un coeficiente de determinación cercano a uno, y esa cifra espectacular no significa nada, porque el modelo solo ha aprendido que el precio de mañana se parece al de hoy.

Por eso el objetivo del sistema es el **rendimiento futuro** a 5, 21 y 63 sesiones, aproximadamente una semana, un mes y un trimestre. El rendimiento es aproximadamente estacionario, comparable entre activos y difícil de predecir, que es exactamente lo que queremos medir. El precio esperado se calcula después, solo para facilitar la lectura.

### 3.2 Validación: walk-forward con purga

No se utiliza en ningún punto una partición aleatoria. Se emplea una ventana expansiva: se entrena con toda la historia disponible, se evalúa sobre el año siguiente y se avanza. Quince pliegues por cada combinación de activo y horizonte.

El detalle que suele pasarse por alto: **entre el final del entrenamiento y el inicio de la prueba se eliminan `h` sesiones**. Sin esa purga, las últimas filas de entrenamiento tienen un rendimiento objetivo que se solapa con el periodo de prueba, y el modelo habría visto parcialmente el futuro que se le pide predecir. Un corte temporal sin purga no evita la fuga.

> **[FIGURA 2]** `fig02_walkforward.png` — Esquema de los quince pliegues con la purga señalada.

### 3.3 Verificación automática de causalidad

Afirmar que no hay fuga de información es fácil; demostrarlo, no. El sistema incluye seis comprobaciones que detienen la ejecución si fallan. La más relevante es la **prueba point-in-time**: se seleccionan fechas al azar, se recalculan todas las variables usando exclusivamente la serie truncada en esa fecha y se exige coincidencia exacta con la tabla de producción. Si alguna variable estuviera mirando al futuro, el valor recalculado diferiría.

Las seis comprobaciones se superan en cada ejecución. El resto de verificaciones cubre unicidad de claves, ausencia de valores faltantes, consistencia de los objetivos mediante búsqueda directa del precio en `t+h`, la identidad de conversión a pesos y la ausencia de objetivo en las últimas sesiones, donde el futuro todavía no existe.

### 3.4 Variables

Veintinueve variables de mercado, agrupadas en cinco familias, todas calculadas únicamente con información disponible al cierre de cada sesión:

| Familia | Ejemplos | Justificación económica |
|---|---|---|
| Rendimientos pasados | 1, 5, 21 y 63 sesiones | momento y reversión a distintas escalas |
| Riesgo realizado | volatilidad a 21 y 63 sesiones, su cociente | régimen de riesgo del activo |
| Tendencia | precio frente a medias de 20, 50 y 200 sesiones | posición relativa, comparable entre activos |
| Contexto | rendimientos de SPY, nivel y anomalía del VIX | estado general del mercado |
| Divisa | rendimientos y volatilidad del USD/MXN | riesgo cambiario |

No se incorporó ningún indicador técnico sin justificación escrita. Diez variables adicionales de sentimiento se describen en la sección 6.

### 3.5 Modelos

Nueve modelos, desde lo trivial hasta lo complejo: rendimiento cero, media histórica, Ridge, ElasticNet, Huber, PLS, LightGBM, LightGBM fuertemente regularizado y XGBoost.

Los hiperparámetros son fijos y conservadores. **No se realizó búsqueda de hiperparámetros**, por dos razones: optimizarlos observando el periodo de prueba es una forma encubierta de fuga, y en un dominio con relación señal-ruido tan baja la búsqueda tiende a seleccionar configuraciones que ajustan ruido.

---

## 4. Resultados predictivos

### 4.1 Ningún modelo supera al modelo nulo en error absoluto

El cociente entre el error absoluto medio de cada modelo y el del modelo que siempre predice cero se sitúa entre 0,97 y 1,16. Es decir, predecir sistemáticamente "no habrá cambio" produce un error comparable o menor que cualquier modelo entrenado.

Este resultado, que parece desalentador, es en realidad una lección metodológica de primer orden: **el error cuadrático premia la prudencia**. Cuando la señal es débil, el predictor que no se arriesga minimiza el error. Una métrica que recompensa no decir nada no puede ser la métrica con la que se evalúa una señal de inversión. El trabajo lo demuestra empíricamente en lugar de citarlo.

### 4.2 El acierto direccional es engañoso sin su punto de comparación

El modelo de media histórica alcanza un 69,3 % de acierto en la dirección a 63 sesiones. Parece un resultado notable hasta que se observa qué hace ese modelo: predecir siempre subida. Su acierto no es habilidad, es el **porcentaje de rendimientos positivos del periodo**, que fue del 69,3 %.

| Horizonte | Tasa base | Mejor modelo | Diferencia |
|---|---|---|---|
| 5 sesiones | 57,6 % | 54,1 % | **−3,5 pp** |
| 21 sesiones | 62,3 % | 56,2 % | **−6,1 pp** |
| 63 sesiones | 69,3 % | 60,4 % | **−8,9 pp** |

Todos los modelos aciertan la dirección **peor** que quien simplemente asumiera que el mercado sube. Reportar el acierto direccional sin su tasa base sería, en el mejor de los casos, incompleto.

> **[FIGURA 3]** `fig05_dir_acc_vs_base.png` — Acierto direccional frente a la tasa base.

### 4.3 Dónde sí hay señal

La métrica adecuada para una señal de inversión no es el error ni el acierto direccional, sino la **correlación de rangos entre lo predicho y lo observado**, conocida en la industria como *information coefficient*. Mide si el modelo ordena correctamente los resultados, que es lo que realmente explota un inversionista.

| Modelo | 5 sesiones | 21 sesiones | 63 sesiones |
|---|---|---|---|
| PLS | +0,036 | +0,060 | +0,074 |
| ElasticNet | +0,033 | +0,055 | **+0,089** |
| Ridge | +0,031 | +0,056 | +0,088 |
| Huber | +0,030 | +0,046 | +0,083 |
| LightGBM | +0,008 | −0,023 | +0,016 |
| XGBoost | +0,010 | −0,025 | +0,022 |

Dos patrones destacan. La señal **crece con el horizonte**, coherente con la literatura sobre momento a medio plazo. Y la señal es **lineal**: los modelos de árboles, con mucha mayor capacidad, no la encuentran.

> **[FIGURA 4]** `fig03_ic_por_modelo.png` — Capacidad predictiva por modelo y horizonte.

### 4.4 Más capacidad, peor resultado

La comparación más limpia del trabajo enfrenta dos configuraciones del **mismo algoritmo**. LightGBM estándar usa 31 hojas y regularización baja; la versión restringida usa 7 hojas, tasa de aprendizaje tres veces menor y regularización diez veces mayor.

| | Error relativo medio | Capacidad predictiva a 63 sesiones |
|---|---|---|
| LightGBM estándar | 1,082 | +0,016 |
| LightGBM restringido | **1,011** | **+0,022** |

Mismo algoritmo, mismos datos, misma validación. Reducir la capacidad mejora ambos indicadores. En un dominio donde la señal es una fracción pequeña de la varianza total, la capacidad adicional se emplea en modelar ruido. Esta observación fundamenta la decisión de no escalar a arquitecturas profundas (sección 9.2).

### 4.5 El hallazgo central: ningún modelo domina

| | AAPL 63 s. | JPM 63 s. | MSFT 63 s. | MSFT 5 s. |
|---|---|---|---|---|
| ElasticNet | +0,048 | **+0,237** | −0,017 | +0,049 |
| Huber | +0,006 | **+0,248** | −0,007 | +0,048 |
| PLS | **+0,081** | +0,150 | −0,009 | +0,077 |
| LightGBM restringido | +0,049 | +0,104 | −0,088 | **+0,069** |

JPM a 63 sesiones presenta la señal más fuerte del estudio y es claramente lineal. MSFT al mismo horizonte no presenta señal en ningún modelo. Y en MSFT a 5 sesiones el mejor resultado lo obtiene un modelo de árboles, la familia que peor se comporta en el resto de casos.

**La premisa de la hipótesis queda respaldada empíricamente.** No es una suposición razonable sino un patrón medido: el modelo apropiado depende del activo y del horizonte.

> **[FIGURA 5]** `fig04_heterogeneidad.png` — Mapa de capacidad predictiva por activo, horizonte y modelo.

**Cautela estadística.** Con objetivos solapados a 63 sesiones, el número de observaciones independientes es muy inferior al número de filas: aproximadamente sesenta frente a 3.780. El valor de 0,248 de JPM debe interpretarse como indicio prometedor, no como resultado robusto.

---

## 5. ¿Aportan las noticias?

### 5.1 Cómo se incorporó el sentimiento

Se recuperaron 55.918 titulares únicos entre 2015 y 2026 para los tres activos. Cada titular se clasificó con **FinBERT**, un modelo de lenguaje afinado sobre textos financieros, obteniendo probabilidades de las clases positiva, neutra y negativa. La distribución resultante —59 % neutra, 21 % positiva, 21 % negativa— es la esperable en cobertura de agencia, mayoritariamente informativa.

La decisión metodológica crítica es **a qué sesión pertenece cada noticia**. Un titular publicado después del cierre no puede informar una predicción que conceptualmente se realiza al cierre. Por tanto: los titulares anteriores a las 16:00 de Nueva York se asignan a esa sesión, y los posteriores, junto con los de fin de semana y festivo, a la sesión siguiente. La verificación confirma que **ningún titular posterior al cierre queda asignado al mismo día**.

La cobertura resultó muy desigual:

| Activo | Sesiones con noticias | Titulares por sesión |
|---|---|---|
| AAPL | 99,4 % | 11,9 |
| MSFT | 92,4 % | 7,0 |
| JPM | 73,9 % | 2,2 |

> **[FIGURA 6]** `fig12_sentimiento.png` — Evolución del sentimiento y de la cobertura informativa.

### 5.2 El experimento

Se comparan dos configuraciones entrenadas sobre la **misma ventana temporal y los mismos pliegues**, cambiando únicamente el conjunto de variables. Comparar el experimento de mercado completo contra el de sentimiento, más corto, habría atribuido al sentimiento una diferencia originada en el periodo.

**Resultado agregado: el sentimiento mejora la capacidad predictiva en 17 de 45 comparaciones, un 38 %.** Por debajo del 50 % que correspondería al azar. En agregado, no aporta.

### 5.3 Pero el agregado esconde la estructura

| Activo | Casos con mejora | Cobertura |
|---|---|---|
| **AAPL** | **12 de 15 (80 %)** | 99,4 %, 11,9 titulares/sesión |
| JPM | 4 de 15 (27 %) | 73,9 %, 2,2 titulares/sesión |
| MSFT | 2 de 15 (13 %) | 92,4 %, 7,0 titulares/sesión |

En AAPL a 5 sesiones el sentimiento mejora el resultado en los cinco modelos. En MSFT a 63 sesiones lo empeora en los cinco.

La explicación más plausible es la **densidad de cobertura**: cuando un activo recibe pocas noticias, la mayoría de las variables de sentimiento son ceros y solo añaden dimensionalidad al problema. AAPL recibe cinco veces más titulares que JPM. Conviene señalar que la explicación no está aislada experimentalmente: MSFT tiene cobertura razonable y aun así empeora.

Y obsérvese que es **el mismo patrón de la sección 4.5**: tampoco existe un conjunto de variables universalmente mejor.

> **[FIGURA 7]** `fig06_efecto_sentimiento.png` — Efecto del sentimiento por activo, horizonte y modelo.

---

## 6. El ensemble dinámico

### 6.1 Diseño

Para cada activo, horizonte y pliegue, los pesos se estiman **exclusivamente con el desempeño observado en pliegues anteriores**. El primer pliegue, sin historia, arranca con pesos iguales. Se contrastan cuatro esquemas de ponderación y dos ventanas de memoria, y se incluye deliberadamente la media simple como referencia: si la adaptación no supera a promediar a ciegas, no aporta nada.

Una consecuencia del diseño merece mención: el ensemble combina predicciones ya generadas fuera de muestra, sin reentrenar. Eso lo hace instantáneo y auditable línea a línea.

### 6.2 La adaptación funciona

| Esquema | Capacidad predictiva media |
|---|---|
| Selección del mejor reciente, ventana corta | **+0,048** |
| Ponderación por capacidad reciente, ventana corta | +0,047 |
| Ponderación por capacidad reciente, ventana larga | +0,042 |
| **Media simple (referencia)** | **+0,039** |
| Ponderación por error inverso | +0,038 |

**Los esquemas adaptativos superan a la media simple**, con una mejora del 18 % para el mejor de ellos. Además emergen dos regularidades:

**La memoria corta gana siempre a la larga.** Usar tres pliegues supera a usar toda la historia en los dos esquemas donde ambas variantes existen. Interpretación de negocio: la ventaja relativa entre modelos no persiste durante años, de modo que promediar quince años de desempeño diluye información útil.

**La selección dura supera a la ponderación suave.** Con cinco candidatos de los cuales dos carecen de señal, repartir peso entre todos arrastra el resultado.

> **[FIGURA 8]** `fig07_ensemble.png` — Comparación de esquemas de ensemble.

### 6.3 Lo que el ensemble no consigue

El mejor ensemble alcanza +0,048 frente a los **+0,059 de ElasticNet por sí solo**. El ensemble no logra desprenderse de sus candidatos débiles con la rapidez suficiente.

La conclusión honesta es matizada: **la adaptación aporta frente a promediar, pero no sustituye a una buena selección previa de familia de modelos**. Con este conjunto de candidatos, un ElasticNet estático habría sido mejor elección. Conviene precisar que "el mejor modelo individual" se identifica observando el resultado final, información de la que nadie dispondría al decidir.

Los parámetros del ensemble se fijaron antes de observar estos resultados y **no se reajustaron** posteriormente.

### 6.4 El ensemble descubre por sí solo dónde usar el sentimiento

Se repitió el ejercicio ampliando los candidatos a diez: cada modelo con y sin variables de sentimiento. Peso total asignado a los candidatos con sentimiento:

| | 5 sesiones | 21 sesiones | 63 sesiones |
|---|---|---|---|
| **AAPL** | **60 %** | **52 %** | 43 % |
| JPM | 46 % | 49 % | 44 % |
| MSFT | 44 % | 44 % | 47 % |

Compárese con la sección 5.3. **El sistema asigna mayoría de peso al sentimiento exactamente en AAPL**, el activo donde el análisis independiente mostró que aporta, y se lo retira a MSFT en los tres horizontes. Reproduce incluso el matiz de que en AAPL a 63 sesiones el sentimiento deja de ayudar.

Lo relevante es que esa decisión se toma **sin observar en ningún momento el periodo que se predice**. Es la demostración de que el mecanismo adaptativo funciona como se pretendía.

> **[FIGURA 9]** `fig08_peso_sentimiento.png` — Peso asignado autónomamente al sentimiento.

---

## 7. De la predicción a la decisión

### 7.1 Señales

Un rendimiento esperado no es una recomendación. La conversión emplea un **umbral adaptado al régimen de riesgo**: `0,25 × volatilidad reciente × √horizonte`. Un 1 % esperado no significa lo mismo en un activo con volatilidad anual del 15 % que en otro del 45 %. Se exige además **acuerdo entre al menos el 60 % de los modelos** sobre el signo.

### 7.2 La señal ordena correctamente los resultados

| Horizonte | Señal | Rendimiento medio posterior | Resultados positivos |
|---|---|---|---|
| 5 sesiones | **COMPRAR** | **+0,67 %** | 58,9 % |
| | MANTENER | +0,35 % | 57,2 % |
| | VENDER | +0,24 % | 55,2 % |
| 21 sesiones | **COMPRAR** | **+2,18 %** | 63,7 % |
| | MANTENER | +1,66 % | 61,2 % |
| | VENDER | +1,31 % | 60,5 % |

Orden perfecto en ambos indicadores y en ambos horizontes. **Cuando el sistema recomienda comprar, el rendimiento medio posterior casi triplica al de una recomendación de venta.** Es la validación independiente de todo el trabajo: la capacidad predictiva medida en la sección 4 se manifiesta en decisiones concretas.

A 63 sesiones el orden se rompe. El diagnóstico es identificable: el 85 % de las señales de AAPL a ese horizonte son de compra y la exposición alcanza el 90 %. El umbral, calibrado con la raíz del horizonte, resulta insuficiente frente a la magnitud de los rendimientos trimestrales, y la señal deja de discriminar. Es una limitación del diseño del umbral.

> **[FIGURA 10]** `fig09_calidad_senal.png` — Calidad de la señal por categoría.

### 7.3 El sistema no anticipa las pérdidas severas

Cabría esperar que una señal de venta anticipara caídas. Los datos dicen que no:

| Horizonte | Probabilidad de pérdida severa tras VENDER, relativa a COMPRAR |
|---|---|
| 5 sesiones | 1,23× |
| 21 sesiones | **0,84×** |
| 63 sesiones | **0,85×** |

A 21 y 63 sesiones una pérdida severa es **menos** probable tras una señal de venta. La pérdida media en el 5 % de peores casos apenas se mueve entre categorías: −13,2 % tras comprar frente a −15,7 % tras vender a 21 sesiones.

**El sistema captura información sobre el centro de la distribución de rendimientos, no sobre sus extremos.** Para un inversionista esto significa que puede ayudar a priorizar entre oportunidades, pero **no sustituye a una gestión de riesgo explícita**.

> **[FIGURA 11]** `fig13_cola_izquierda.png` — Riesgo de cola condicionado a la señal.

### 7.4 Comparación con las referencias

| Activo | Comprar y mantener | Señales | Solo posiciones largas | SPY |
|---|---|---|---|---|
| AAPL | **2.499 %** | 439 % | 616 % | 613 % |
| MSFT | **2.031 %** | 246 % | 298 % | 613 % |
| JPM | **951 %** | 126 % | 364 % | 613 % |

Ninguna estrategia basada en las señales supera a comprar y mantener. Tres mecanismos lo explican:

**Estar fuera del mercado cuesta.** La exposición oscila entre el 30 % y el 90 %; cada sesión sin posición en un mercado alcista es rendimiento perdido.

**Las posiciones cortas destruyen valor.** El rendimiento medio tras una señal de venta sigue siendo positivo. Por eso la variante que solo toma posiciones largas supera sistemáticamente a la que también vende en corto.

**Ordenar bien no equivale a ganar más.** La diferencia entre categorías es de décimas de punto porcentual; la deriva alcista del periodo es de varios puntos.

> **[FIGURA 12]** `fig10_equity.png` — Evolución del capital frente a las referencias.

---

## 8. La perspectiva del inversionista mexicano

El rendimiento que percibe quien consume en pesos combina dos fuentes:

`(1 + rendimiento en pesos) = (1 + rendimiento del activo) × (1 + variación del tipo de cambio)`

Para la **expectativa** se asume paseo aleatorio en la divisa. No es una simplificación por comodidad: desde Meese y Rogoff (1983) está documentado que ningún modelo supera sistemáticamente al paseo aleatorio en predicción cambiaria a estos horizontes. En consecuencia, el rendimiento esperado en pesos iguala al esperado en dólares.

El hallazgo está en el **análisis de lo ocurrido**:

| Horizonte | Rendimiento medio USD | Rendimiento medio MXN | Casos que cambian de signo |
|---|---|---|---|
| 5 sesiones | +0,44 % | +0,49 % | **15,7 %** |
| 21 sesiones | +1,85 % | +2,05 % | **13,9 %** |
| 63 sesiones | +5,58 % | +6,22 % | **12,8 %** |

**Aproximadamente uno de cada siete resultados cambia de signo al convertirse a pesos.** Una inversión rentable en dólares puede ser una pérdida en pesos, y viceversa. En el periodo analizado la divisa aportó rendimiento positivo, porque el peso se depreció frente al dólar, pero esa aportación es una apuesta cambiaria no remunerada por riesgo, no una fuente estable de rentabilidad.

La respuesta a la pregunta de investigación es afirmativa: **la perspectiva cambia de forma material**, y cualquier sistema dirigido a un inversionista mexicano que ignore la divisa está omitiendo un factor que altera el resultado en un caso de cada siete.

> **[FIGURA 13]** `fig11_divisa.png` — Descomposición del rendimiento en activo y divisa.

---

## 9. Conclusiones

### 9.1 Qué se ha demostrado

**Existe información predictiva, pequeña y medible.** Los modelos lineales regularizados alcanzan correlaciones de rangos de hasta +0,089, y la señal se traduce en decisiones útiles: una recomendación de compra va seguida de rendimientos que casi triplican los de una recomendación de venta.

**La adaptación supera a la rigidez, con matices.** Los esquemas adaptativos mejoran un 18 % sobre la media simple, la memoria corta supera a la larga y el ensemble identifica de forma autónoma en qué activo conviene incorporar sentimiento. Pero no supera al mejor modelo individual identificado a posteriori.

**La complejidad no ayuda en este dominio.** Reducir la capacidad del mismo algoritmo mejora simultáneamente el error y la capacidad predictiva.

**El sistema no supera al mercado.** Ninguna estrategia basada en las señales bate a comprar y mantener, y el sistema no anticipa las pérdidas severas.

**La divisa altera materialmente el resultado** para un inversionista mexicano.

### 9.2 Limitaciones

1. **Universo reducido y sesgado por la supervivencia.** Tres empresas de gran éxito en el periodo. El sistema nunca se enfrentó a un activo en declive estructural, por lo que los resultados sobre riesgo de cola no son extrapolables a ese caso.
2. **Ventana reducida en el experimento de sentimiento.** Seis pliegues frente a quince, con la consiguiente pérdida de potencia estadística.
3. **Objetivos solapados.** A 63 sesiones el número de observaciones independientes es muy inferior al de filas, lo que amplía los intervalos de confianza.
4. **Ausencia de costes de transacción.** Su inclusión reduciría el rendimiento de las estrategias con rotación alta, ampliando la ventaja de comprar y mantener.
5. **Periodo mayoritariamente alcista.** Entre el 69 % y el 70 % de los rendimientos a 63 sesiones fueron positivos, lo que condiciona todas las comparaciones.
6. **El sentimiento se limita a titulares**, no al cuerpo de las noticias, y solo a una fuente.

### 9.3 Líneas futuras

La dirección natural **no** es incorporar arquitecturas de mayor capacidad. La evidencia de la sección 4.4 apunta al contrario: en este régimen de señal-ruido, la capacidad adicional degrada el desempeño. Los modelos recurrentes y los transformadores para series temporales se descartaron por esta razón, no por limitaciones de tiempo.

Las direcciones con mayor recorrido son: ampliar el universo para permitir modelos de sección cruzada y neutralizar el sesgo de supervivencia; incorporar costes de transacción; combinar el sistema con una capa explícita de gestión de riesgo que cubra la debilidad identificada en la sección 7.3; y explorar la cobertura cambiaria. En el plano de la ingeniería, una versión productiva migraría a un lago de datos con orquestación, registro de modelos, servicio de inferencia y monitorización de deriva.

---

## 10. Reproducibilidad y aplicación

El proyecto completo se regenera desde las fuentes originales con un único comando, en menos de tres minutos sobre un equipo portátil. Cada ejecución repite las seis comprobaciones de integridad temporal antes de modelizar.

Se desarrolló además una **aplicación funcional** que permite seleccionar activo y horizonte y consulta los artefactos del sistema para devolver el precio actual, el rendimiento y el precio esperados, la doble lectura en dólares y pesos, el sentimiento reciente, la señal y los pesos vigentes del ensemble. Su propósito es demostrar que el sistema es utilizable, no operable con dinero real.

> **Advertencia.** Los resultados y señales de este trabajo tienen finalidad exclusivamente académica y experimental. No constituyen asesoramiento financiero.

---

## Bibliografía

- Fama, E. F. (1970). Efficient capital markets: a review of theory and empirical work. *Journal of Finance*, 25(2).
- Meese, R. y Rogoff, K. (1983). Empirical exchange rate models of the seventies. *Journal of International Economics*, 14(1).
- Jegadeesh, N. y Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1).
- Araci, D. (2019). FinBERT: financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*.
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS*.
- Chen, T. y Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Grinold, R. y Kahn, R. (1999). *Active Portfolio Management*. McGraw-Hill.

---

## Anexos

- **A. Código fuente.** Repositorio con el pipeline completo, el protocolo de validación y la aplicación.
- **B. Análisis exploratorio ampliado.**
- **C. Tablas completas de experimentos.** Métricas por activo, horizonte y modelo.
- **D. Registro de decisiones.** Diecinueve decisiones metodológicas con su motivo, alternativas descartadas e impacto.
- **E. Diccionario de variables.**
