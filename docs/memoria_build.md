# Sistema adaptativo multimodal para la predicción de rendimientos bursátiles mediante ensemble dinámico y análisis de sentimiento
Trabajo Fin de Máster · Máster en Big Data, Data Science e Inteligencia Artificial · Universidad Complutense de Madrid

Autor: Xavier Gutiérrez Palma · Tutores: Carlos Ortega y Santiago Mota

---

## 1. El problema y por qué importa

Un inversionista mexicano que compra acciones estadounidenses toma decisiones sobre el activo, el horizonte y una exposición adicional al tipo de cambio. En este trabajo no parto de la idea de que el mercado pueda predecirse de forma consistente. El objetivo es más limitado: medir si existe una señal aprovechable en datos públicos y comprobar si esa señal cambia según el activo y el horizonte.

A partir de ese planteamiento se construyó un sistema que combina información de mercado, noticias financieras y USD/MXN, y que adapta la combinación de modelos en función de su desempeño fuera de muestra.

### 1.1 Hipótesis

> Dado que el comportamiento financiero puede variar entre activos y horizontes, un ensemble que actualice sus pesos con el desempeño reciente de los modelos puede superar a una combinación estática.

La hipótesis se limita a la forma de combinar los modelos. La capacidad de superar o no al mercado se evalúa por separado en el backtest financiero.

### 1.2 Qué aporta este trabajo

1. Un conjunto de datos construido, no descargado: precios, volatilidad implícita, tipo de cambio y 55.918 titulares procesados con un modelo de lenguaje financiero, alineados temporalmente sin fugas de información.

2. Un protocolo de validación con purga del horizonte y una prueba automática de causalidad que verifica, sobre los propios datos, que ninguna variable usa información futura.

3. Un ensemble dinámico cuyos pesos se estiman exclusivamente con observaciones anteriores a la fecha que se predice.

4. La descomposición del rendimiento en activo y divisa desde la perspectiva de un inversionista que consume en pesos.

5. Una aplicación funcional que consume el sistema y devuelve predicciones y señales.

### 1.3 Lo que el lector encontrará

El resultado general es menos espectacular que la idea inicial del proyecto, pero más útil para evaluarlo con rigor: se detecta una señal predictiva pequeña, aunque las estrategias construidas a partir de ella no superan a comprar y mantener. El análisis se apoya en 306.180 predicciones fuera de muestra. Esta diferencia entre capacidad predictiva y rentabilidad será importante en las secciones posteriores.

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

Se eligieron tres empresas de sectores distintos (tecnología de consumo, software empresarial y banca) para evitar que toda la comparación dependiera de un solo sector. El periodo analizado, de enero de 2006 a septiembre de 2026, incluye episodios con comportamientos de mercado muy diferentes, entre ellos la crisis financiera de 2008, la caída y recuperación de 2020 y el ciclo de tipos posterior. Esto permite observar si el desempeño relativo de los modelos se mantiene o cambia a lo largo del tiempo.

Volumen final: 15.612 filas con 29 variables para el análisis de mercado; 8.817 filas con 39 variables para el análisis con sentimiento.

![Figura 1. Precios normalizados, VIX y USD/MXN, 2006-2026.](outputs/figures/fig01_panel_contexto.png)

### 2.2 Derechos de uso

Los datos de mercado se obtienen mediante la librería abierta `yfinance` para uso académico y no comercial. Los titulares proceden de la API de Alpaca bajo cuenta gratuita. No se redistribuyen los datos brutos: el repositorio contiene el código de ingesta, que los regenera de forma determinista, y un manifiesto con fechas, número de filas y momento de descarga de cada serie.

### 2.3 Arquitectura Bronze / Silver / Gold

- Bronze: descargas crudas, una por símbolo, sin transformación alguna. Capa inmutable.

- Silver: limpieza y alineación temporal. El calendario maestro son las sesiones de SPY; VIX y USD/MXN se reindexan a él mediante arrastre hacia adelante limitado a cinco días. El arrastre solo mira hacia atrás, de modo que es causal por construcción. La interpolación, alternativa natural, habría usado valores futuros e introducido una fuga silenciosa.

- Gold: variables explicativas y objetivos. Es la única capa que consumen los modelos.

---

## 3. Metodología

Esta sección describe las decisiones metodológicas que más afectan la validez de los resultados: la definición del objetivo, la separación temporal, la construcción de variables y los controles contra fuga de información.

### 3.1 Predecir rendimiento, no precio

Los precios presentan una fuerte dependencia temporal y una tendencia de largo plazo. Por esa razón, una métrica alta al predecir directamente el precio puede ser engañosa: una parte importante del resultado puede explicarse simplemente porque el precio futuro suele estar cerca del precio actual.

El objetivo del sistema es, por tanto, el rendimiento futuro a 5, 21 y 63 sesiones, aproximadamente una semana, un mes y un trimestre. Trabajar con rendimientos facilita la comparación entre activos y plantea un problema predictivo más exigente. El precio esperado se reconstruye después únicamente para presentar los resultados de forma más intuitiva.

### 3.2 Validación: walk-forward con purga

No se utiliza en ningún punto una partición aleatoria. Se emplea una ventana expansiva: se entrena con toda la historia disponible, se evalúa sobre el año siguiente y se avanza. Quince pliegues por cada combinación de activo y horizonte.

Entre el final del entrenamiento y el inicio de la prueba se eliminan `h` sesiones. Sin esta purga, las últimas filas de entrenamiento tendrían un rendimiento objetivo que se solapa con el periodo de prueba. En ese caso existiría fuga de información aunque la partición respetara el orden temporal.

![Figura 2. Esquema de los quince pliegues con la purga señalada.](outputs/figures/fig02_walkforward.png)

### 3.3 Verificación automática de causalidad

El pipeline incluye seis comprobaciones automáticas orientadas a detectar fugas o inconsistencias antes de entrenar. La principal es una prueba point-in-time: se seleccionan fechas al azar, se recalculan las variables utilizando la serie truncada en cada fecha y se comparan con la tabla de producción. Una diferencia entre ambos cálculos indicaría que alguna transformación utilizó información posterior.

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

No se incorporaron indicadores técnicos únicamente por disponibilidad; cada variable incluida tiene una justificación asociada. Las diez variables adicionales de sentimiento se describen en la sección 5.

### 3.5 Modelos

Se evaluaron nueve alternativas: rendimiento cero, media histórica, Ridge, ElasticNet, Huber, PLS, LightGBM, una versión más regularizada de LightGBM y XGBoost.

Los hiperparámetros se fijaron antes de comparar los resultados de prueba y se mantuvieron sin cambios entre pliegues. No se realizó una búsqueda extensa de hiperparámetros. Hacer ajustes utilizando el desempeño del periodo de prueba habría sesgado la evaluación, y además habría aumentado el riesgo de seleccionar configuraciones demasiado adaptadas a una muestra con una relación señal-ruido baja.

---

## 4. Resultados predictivos
### 4.1 Comportamiento del error absoluto frente al modelo nulo

Ningún modelo entrenado sobre las variables de mercado supera de forma consistente al modelo que siempre predice cero: los cocientes de error absoluto medio se sitúan entre 1,00 y 1,16. La única excepción es la media histórica, que lo mejora en las nueve combinaciones con cocientes de 0,90 a 0,99. La sección 4.2 muestra por qué esa mejora no implica capacidad predictiva.

Este resultado muestra por qué el MAE no es suficiente para evaluar una señal de inversión. Cuando los retornos son pequeños y ruidosos, predecir valores cercanos a cero puede reducir el error absoluto sin aportar capacidad para ordenar oportunidades. Por esa razón el análisis se complementa con métricas de dirección y con el information coefficient.

### 4.2 El acierto direccional es engañoso sin su punto de comparación

El modelo de media histórica alcanza un 69,3 % de acierto en la dirección a 63 sesiones. Parece un resultado notable hasta que se observa qué hace ese modelo: predecir siempre subida. Su acierto no es habilidad, es el porcentaje de rendimientos positivos del periodo, que fue del 69,3 %.

| Horizonte | Tasa base | Mejor modelo | Diferencia |
|---|---|---|---|
| 5 sesiones | 57,6 % | 54,1 % | −3,5 pp |
| 21 sesiones | 62,3 % | 56,2 % | −6,1 pp |
| 63 sesiones | 69,3 % | 60,4 % | −8,9 pp |

Todos los modelos aciertan la dirección peor que quien simplemente asumiera que el mercado sube. Reportar el acierto direccional sin su tasa base sería, en el mejor de los casos, incompleto.

![Figura 3. Acierto direccional frente a la tasa base.](outputs/figures/fig05_dir_acc_vs_base.png)

### 4.3 Dónde sí hay señal

Para medir si las predicciones permiten ordenar oportunidades se utiliza además la correlación de rangos entre lo predicho y lo observado, denominada *information coefficient* (IC). Esta métrica no exige acertar el nivel exacto del rendimiento; evalúa si las observaciones con mayor predicción tienden también a presentar mayores rendimientos realizados.

| Modelo | 5 sesiones | 21 sesiones | 63 sesiones |
|---|---|---|---|
| PLS | +0,036 | +0,060 | +0,074 |
| ElasticNet | +0,033 | +0,055 | +0,089 |
| Ridge | +0,031 | +0,056 | +0,088 |
| Huber | +0,030 | +0,046 | +0,083 |
| LightGBM | +0,009 | −0,024 | +0,014 |
| XGBoost | +0,009 | −0,023 | +0,020 |

En la muestra aparecen dos patrones. El IC medio aumenta con el horizonte, un resultado compatible con evidencia de momentum a medio plazo (Jegadeesh y Titman, 1993). Además, los modelos lineales regularizados obtienen mejores resultados que los modelos de árboles en la mayoría de las comparaciones.

![Figura 4. Capacidad predictiva por modelo y horizonte.](outputs/figures/fig03_ic_por_modelo.png)

### 4.4 Efecto de la capacidad del modelo

La comparación más limpia del trabajo enfrenta dos configuraciones del mismo algoritmo. LightGBM estándar usa 31 hojas y regularización baja; la versión restringida usa 7 hojas, tasa de aprendizaje tres veces menor y regularización diez veces mayor.

| | Error relativo medio | Capacidad predictiva a 63 sesiones |
|---|---|---|
| LightGBM estándar | 1,080 | +0,014 |
| LightGBM restringido | 1,009 | +0,022 |

Con los mismos datos y el mismo esquema de validación, la configuración más restringida mejora ambos indicadores. En este experimento, la mayor capacidad de LightGBM no se traduce en una mejor generalización y el resultado es compatible con sobreajuste. Esta observación se utiliza más adelante para justificar por qué no se priorizó aumentar la complejidad del modelo.

### 4.5 Resultados por activo y horizonte

| | AAPL 63 s. | JPM 63 s. | MSFT 63 s. | MSFT 5 s. |
|---|---|---|---|---|
| ElasticNet | +0,048 | +0,237 | −0,017 | +0,049 |
| Huber | +0,006 | +0,248 | −0,007 | +0,048 |
| PLS | +0,081 | +0,150 | −0,009 | +0,077 |
| LightGBM restringido | +0,049 | +0,105 | −0,089 | +0,069 |

JPM a 63 sesiones presenta la señal más fuerte del estudio y es claramente lineal. MSFT al mismo horizonte no presenta señal en ningún modelo. Y en MSFT a 5 sesiones el mejor resultado lo obtiene un modelo de árboles, la familia que peor se comporta en el resto de casos.

Dentro del universo estudiado, estos resultados apoyan la hipótesis de que el desempeño relativo de los modelos depende del activo y del horizonte. La conclusión no se extiende fuera de estos tres activos sin una validación adicional.

![Figura 5. Mapa de capacidad predictiva por activo, horizonte y modelo.](outputs/figures/fig04_heterogeneidad.png)

Cautela estadística. Con objetivos solapados a 63 sesiones, el número de observaciones independientes es muy inferior al número de filas: aproximadamente sesenta frente a 3.780. El valor de 0,248 de JPM debe interpretarse como indicio prometedor, no como resultado robusto.

---

## 5. ¿Aportan las noticias?
### 5.1 Cómo se incorporó el sentimiento

Se recuperaron 55.918 titulares únicos entre 2015 y 2026 para los tres activos. Cada titular se clasificó con FinBERT, un modelo de lenguaje afinado sobre textos financieros, obteniendo probabilidades para las clases positiva, neutra y negativa. La clase neutra fue la más frecuente: 59 % de los titulares, frente a 21 % positivos y 21 % negativos.

Para evitar fuga temporal fue necesario definir a qué sesión pertenece cada noticia. Los titulares publicados antes de las 16:00 de Nueva York se asignan a esa sesión; los publicados después del cierre, durante fines de semana o en festivos se asignan a la siguiente sesión disponible. Una comprobación automática verifica que ningún titular posterior al cierre quede asociado al mismo día bursátil.

La cobertura resultó muy desigual:

| Activo | Sesiones con noticias | Titulares por sesión |
|---|---|---|
| AAPL | 99,4 % | 11,9 |
| MSFT | 92,4 % | 7,0 |
| JPM | 73,9 % | 2,2 |

![Figura 6. Evolución del sentimiento y de la cobertura informativa.](outputs/figures/fig12_sentimiento.png)

### 5.2 El experimento

Se comparan dos configuraciones entrenadas sobre la misma ventana temporal y los mismos pliegues, cambiando únicamente el conjunto de variables. Comparar el experimento de mercado completo contra el de sentimiento, más corto, habría atribuido al sentimiento una diferencia originada en el periodo.

En 16 de 45 comparaciones (36 %) el IC mejora al añadir variables de sentimiento. Por tanto, no aparece una mejora agregada consistente frente al conjunto de variables de mercado.

### 5.3 Resultados por activo

| Activo | Casos con mejora | Cobertura |
|---|---|---|
| AAPL | 12 de 15 (80 %) | 99,4 %, 11,9 titulares/sesión |
| JPM | 2 de 15 (13 %) | 73,9 %, 2,2 titulares/sesión |
| MSFT | 2 de 15 (13 %) | 92,4 %, 7,0 titulares/sesión |

En AAPL a 5 sesiones el sentimiento mejora el resultado en los cinco modelos. En MSFT a 63 sesiones lo empeora en los cinco.

Una posible explicación es la densidad de cobertura. Cuando un activo recibe pocas noticias, muchas variables agregadas de sentimiento toman valores cercanos a cero y pueden añadir dimensionalidad sin información suficiente. AAPL recibe alrededor de cinco veces más titulares por sesión que JPM. Sin embargo, esta explicación no queda aislada por el experimento: JPM y MSFT obtienen el mismo número de casos con mejora pese a tener coberturas muy distintas.

El resultado vuelve a mostrar heterogeneidad entre activos: el mismo conjunto de variables no mejora de forma uniforme los tres casos.

![Figura 7. Efecto del sentimiento por activo, horizonte y modelo.](outputs/figures/fig06_efecto_sentimiento.png)

---

## 6. El ensemble dinámico
### 6.1 Diseño

Para cada activo, horizonte y pliegue, los pesos se estiman exclusivamente con el desempeño observado en pliegues anteriores. El primer pliegue, al no disponer todavía de historia, utiliza pesos iguales. Se comparan cuatro esquemas de ponderación y dos ventanas de memoria. La media simple se utiliza como referencia para comprobar si la adaptación añade valor frente a una combinación no adaptativa.

El ensemble trabaja sobre predicciones que ya fueron generadas fuera de muestra. Por tanto, no requiere volver a entrenar los modelos para probar cada esquema de combinación y permite rastrear el origen de cada predicción.

### 6.2 Resultados del ensemble

| Esquema | Capacidad predictiva media |
|---|---|
| Ponderación por capacidad reciente, ventana corta | +0,047 |
| Selección del mejor reciente, ventana corta | +0,044 |
| Ponderación por capacidad reciente, ventana larga | +0,042 |
| Media simple (referencia) | +0,039 |
| Ponderación por error inverso, ventana corta | +0,038 |

Los esquemas adaptativos superan a la media simple en este experimento; el mejor alcanza un IC medio un 18 % mayor que la referencia.

En los dos esquemas en los que se compararon ambas ventanas, utilizar los tres pliegues más recientes dio mejor resultado que usar toda la historia. Esto sugiere que el desempeño relativo de los modelos cambia con el tiempo y que una memoria demasiado larga puede diluir información reciente.

![Figura 8. Comparación de esquemas de ensemble.](outputs/figures/fig07_ensemble.png)

### 6.3 Límites del ensemble

El mejor ensemble alcanza +0,047 frente a los +0,059 de ElasticNet por sí solo. El ensemble no logra desprenderse de sus candidatos débiles con la rapidez suficiente.

La adaptación mejora frente a la media simple, pero no supera a ElasticNet en el resultado agregado. Con este conjunto de candidatos, seleccionar ElasticNet de forma retrospectiva habría producido un IC mayor. Esa comparación debe interpretarse con cuidado: identificar al mejor modelo individual requiere observar todo el periodo, información que no estaba disponible en cada fecha del backtest.

Los parámetros del ensemble se fijaron antes de observar estos resultados y no se reajustaron posteriormente.

### 6.4 Peso asignado a las variables de sentimiento

Se repitió el ejercicio ampliando los candidatos a diez: cada modelo con y sin variables de sentimiento. Peso total asignado a los candidatos con sentimiento:

| | 5 sesiones | 21 sesiones | 63 sesiones |
|---|---|---|---|
| AAPL | 61 % | 51 % | 42 % |
| JPM | 44 % | 50 % | 41 % |
| MSFT | 45 % | 44 % | 48 % |

El patrón es consistente con los resultados de la sección 5.3. AAPL recibe el mayor peso en candidatos con sentimiento a 5 y 21 sesiones, que son los casos donde su incorporación fue más favorable. En MSFT el peso permanece por debajo del 50 % en los tres horizontes.

La asignación se calcula únicamente con pliegues anteriores. Por tanto, este comportamiento surge sin utilizar el periodo que se está prediciendo.

![Figura 9. Peso asignado autónomamente al sentimiento.](outputs/figures/fig08_peso_sentimiento.png)

---

## 7. De la predicción a la decisión
### 7.1 Señales

Un rendimiento esperado no es una recomendación. La conversión emplea un umbral adaptado al régimen de riesgo: $0{,}25 \times \text{volatilidad reciente} \times \sqrt{h}$. Un 1 % esperado no significa lo mismo en un activo con volatilidad anual del 15 % que en otro del 45 %. Se exige además acuerdo entre al menos el 60 % de los modelos sobre el signo.

### 7.2 Resultados de las señales

| Horizonte | Señal | Rendimiento medio posterior | Resultados positivos |
|---|---|---|---|
| 5 sesiones | COMPRAR | +0,66 % | 58,9 % |
| | MANTENER | +0,36 % | 57,2 % |
| | VENDER | +0,25 % | 55,4 % |
| 21 sesiones | COMPRAR | +2,19 % | 63,7 % |
| | MANTENER | +1,65 % | 61,1 % |
| | VENDER | +1,32 % | 60,6 % |

En 5 y 21 sesiones se observa un orden monotónico entre las tres categorías: COMPRAR presenta el mayor rendimiento medio posterior y VENDER el menor. A 5 sesiones la diferencia entre ambas categorías es de 0,41 puntos porcentuales; a 21 sesiones es de 0,87 puntos. Esto muestra que parte de la capacidad de ordenación observada en el IC también aparece al convertir las predicciones en señales discretas.

A 63 sesiones el orden se rompe. El diagnóstico es identificable: el 85 % de las señales de AAPL a ese horizonte son de compra y la exposición alcanza el 90 %. El umbral, calibrado con la raíz del horizonte, resulta insuficiente frente a la magnitud de los rendimientos trimestrales, y la señal deja de discriminar. Es una limitación del diseño del umbral.

![Figura 10. Calidad de la señal por categoría.](outputs/figures/fig09_calidad_senal.png)

### 7.3 El sistema no anticipa las pérdidas severas

Cabría esperar que una señal de venta anticipara caídas. Los datos dicen que no:

| Horizonte | Probabilidad de pérdida severa tras VENDER, relativa a COMPRAR |
|---|---|
| 5 sesiones | 1,21× |
| 21 sesiones | 0,85× |
| 63 sesiones | 0,85× |

A 21 y 63 sesiones una pérdida severa es menos probable tras una señal de venta. La pérdida media en el 5 % de peores casos apenas se mueve entre categorías: −13,3 % tras comprar frente a −15,7 % tras vender a 21 sesiones.

Estos resultados sugieren que la señal contiene más información sobre el comportamiento medio de los rendimientos que sobre sus extremos. En consecuencia, no debería interpretarse como sustituto de una capa específica de gestión de riesgo.

La probabilidad de caer en el decil de peores rendimientos se mantiene en torno al 10 % en las tres categorías a 5 y 21 sesiones, que es el valor esperado si la señal no aportara información sobre la cola de la distribución.

![Figura 11. Riesgo de cola condicionado a la señal.](outputs/figures/fig13_cola_izquierda.png)

### 7.4 Comparación con las referencias

| Activo | Comprar y mantener | Señales | Solo posiciones largas | SPY |
|---|---|---|---|---|
| AAPL | 2.499 % | 439 % | 616 % | 613 % |
| MSFT | 2.031 % | 239 % | 248 % | 613 % |
| JPM | 951 % | 100 % | 339 % | 613 % |

Ninguna estrategia basada en las señales supera a comprar y mantener. Tres mecanismos lo explican:

La menor exposición explica una parte de la diferencia: según el activo y el horizonte, la estrategia permanece invertida entre el 30 % y el 90 % del tiempo. En un periodo mayoritariamente alcista, las sesiones fuera del mercado reducen el rendimiento acumulado.

Las posiciones cortas también reducen el resultado en esta muestra. El rendimiento medio posterior a una señal de venta sigue siendo positivo, por lo que la variante de solo posiciones largas supera a la estrategia que permite posiciones cortas.

Por último, una señal puede ordenar correctamente los rendimientos sin generar una ventaja suficiente para compensar la deriva alcista del mercado. Las diferencias entre categorías son pequeñas frente al rendimiento acumulado de comprar y mantener.

![Figura 12. Evolución del capital frente a las referencias.](outputs/figures/fig10_equity.png)

---

## 8. La perspectiva del inversionista mexicano

El rendimiento que percibe quien consume en pesos combina dos fuentes:

$(1+r_{MXN})=(1+r_{activo})(1+r_{USD/MXN})$

Para la expectativa de USD/MXN se utiliza un paseo aleatorio como referencia, es decir, no se introduce una predicción adicional sobre la divisa. Esta elección es coherente con la literatura clásica sobre la dificultad de superar de forma estable a este benchmark en tipos de cambio (Meese y Rogoff, 1983). En consecuencia, la expectativa puntual en pesos coincide con la expectativa en dólares antes de observar el movimiento futuro de la divisa.

El hallazgo está en el análisis de lo ocurrido:

| Horizonte | Rendimiento medio USD | Rendimiento medio MXN | Casos que cambian de signo |
|---|---|---|---|
| 5 sesiones | +0,44 % | +0,49 % | 15,7 % |
| 21 sesiones | +1,85 % | +2,05 % | 13,9 % |
| 63 sesiones | +5,58 % | +6,22 % | 12,8 % |

Entre el 12,8 % y el 15,7 % de los casos cambia de signo al convertir el rendimiento a pesos. Por tanto, una operación positiva en dólares puede terminar siendo negativa en pesos, y viceversa. Durante el periodo estudiado, la depreciación del peso elevó el rendimiento medio expresado en MXN, pero se trata de una exposición cambiaria adicional y no de una fuente garantizada de rentabilidad.

En esta muestra, incorporar la divisa modifica la interpretación del resultado en aproximadamente uno de cada siete casos, por lo que no resulta equivalente evaluar la inversión únicamente en dólares.

![Figura 13. Descomposición del rendimiento en activo y divisa.](outputs/figures/fig11_divisa.png)

---

## 9. Conclusiones
### 9.1 Resultados principales

Los resultados muestran una señal predictiva pequeña y heterogénea. Los modelos lineales regularizados alcanzan correlaciones de rangos de hasta +0,089 y, en 5 y 21 sesiones, las señales COMPRAR/MANTENER/VENDER ordenan los rendimientos posteriores en la dirección esperada.

El ensemble adaptativo mejora un 18 % frente a la media simple, y las ventanas cortas funcionan mejor que las largas en los esquemas comparables. Sin embargo, el ensemble no supera al mejor modelo individual identificado a posteriori.

En los modelos evaluados, una mayor capacidad no produjo mejores resultados. La versión regularizada de LightGBM generalizó mejor que la configuración estándar, y los modelos lineales fueron superiores en buena parte de las comparaciones.

Las estrategias derivadas de las señales no superan a comprar y mantener y tampoco anticipan de forma consistente las pérdidas severas. La utilidad del sistema se encuentra, por tanto, en la capacidad de ordenar oportunidades, no en sustituir una estrategia de inversión o de gestión de riesgo.

Por último, la conversión a MXN cambia el signo del rendimiento en aproximadamente uno de cada siete casos. Para un inversionista mexicano, la divisa modifica de forma relevante la lectura del resultado.

### 9.2 Limitaciones

1. Universo reducido y sesgado por la supervivencia. Tres empresas de gran éxito en el periodo. El sistema nunca se enfrentó a un activo en declive estructural, por lo que los resultados sobre riesgo de cola no son extrapolables a ese caso.

2. Ventana reducida en el experimento de sentimiento. Seis pliegues frente a quince, con la consiguiente pérdida de potencia estadística.

3. Objetivos solapados. A 63 sesiones el número de observaciones independientes es muy inferior al de filas, lo que amplía los intervalos de confianza.

4. Ausencia de costes de transacción. Su inclusión reduciría el rendimiento de las estrategias con rotación alta, ampliando la ventaja de comprar y mantener.

5. Periodo mayoritariamente alcista. Entre el 69 % y el 70 % de los rendimientos a 63 sesiones fueron positivos, lo que condiciona todas las comparaciones.

6. El sentimiento se limita a titulares, no al cuerpo de las noticias, y solo a una fuente.

### 9.3 Líneas futuras

Los resultados no justifican, por sí solos, aumentar la capacidad de los modelos. La comparación entre las dos configuraciones de LightGBM sugiere que el sobreajuste es un riesgo relevante en este problema. Por ello, dentro del alcance del trabajo se priorizaron modelos más simples y regularizados. La evaluación de arquitecturas recurrentes o transformadores para series temporales queda como trabajo futuro y requeriría un experimento específico.

Las siguientes extensiones serían ampliar el universo de activos para estudiar modelos de sección cruzada y reducir el sesgo de supervivencia, incorporar costes de transacción, añadir una capa explícita de gestión de riesgo y analizar estrategias de cobertura cambiaria. En ingeniería, una versión productiva podría migrar a un lago de datos con orquestación, registro de modelos, servicio de inferencia y monitorización de deriva.

---

## 10. Reproducibilidad y aplicación

El proyecto completo se regenera desde las fuentes originales con un único comando, en menos de tres minutos sobre un equipo portátil. Cada ejecución repite las seis comprobaciones de integridad temporal antes de modelizar.

Se desarrolló además una aplicación funcional que permite seleccionar activo y horizonte y consulta los artefactos del sistema para devolver el precio actual, el rendimiento y el precio esperados, la doble lectura en dólares y pesos, el sentimiento reciente, la señal y los pesos vigentes del ensemble. Su propósito es demostrar que el sistema es utilizable, no operable con dinero real.

> Advertencia. Los resultados y señales de este trabajo tienen finalidad exclusivamente académica y experimental. No constituyen asesoramiento financiero.

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

- A. Código fuente. Repositorio con el pipeline completo, el protocolo de validación y la aplicación.

- B. Análisis exploratorio ampliado.

- C. Tablas completas de experimentos. Métricas por activo, horizonte y modelo.

- D. Registro de decisiones. Diecinueve decisiones metodológicas con su motivo, alternativas descartadas e impacto.

- E. Diccionario de variables.
