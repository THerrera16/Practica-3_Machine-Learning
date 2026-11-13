# Inciso 3: K-Nearest Neighbors (K-NN desde cero)

## Qué se hizo

* **Implementación desde cero** (solo NumPy) de **K-NN** (`KNNDesdeCero`) con distancia euclidiana (vectorizada mediante expansión algebraica) y distancia Manhattan (calculada por bloques de ~64 MB para optimizar memoria), además de votación simple (`uniform`) y ponderada por distancia inversa (`distance`, $w = 1/d$). Los resultados coinciden al 100 % con `KNeighborsClassifier` de `scikit-learn` (usado únicamente para verificación).
* **Datasets:** Sintético (`mdlon_sintetico.csv`, 2000 muestras, 500 rasgos, 2 clases) y Benchmark (`WineQT_benchmark.csv`, 1143 muestras, 11 rasgos, agrupado en 3 clases: baja $\le 5$, media $= 6$, alta $\ge 7$).
* **Validación y búsqueda de hiperparámetros:** Hold-Out estratificado (70 % entrenamiento / 30 % prueba) y K-Fold estratificado ($k = 5$). Se evaluaron 36 combinaciones ($k \in \{1, 3, 5, 7, 9, 11, 15, 21, 31\}$, métrica $\in \{\text{euclidean}, \text{manhattan}\}$, pesos $\in \{\text{uniform}, \text{distance}\}$). La mejor configuración se eligió mediante CV **exclusivamente dentro del 70 % de entrenamiento** para evitar fuga de información (*data leakage*), ajustando `StandardScaler` por separado en cada partición.

## Resultados (exactitud y F1-Score Macro)

| Dataset | Modelo | Hiperparámetros | Hold-Out Acc | Hold-Out F1 Macro | K-Fold ($k=5$) Acc |
|---|---|---|---|---|---|
| Sintético (mdlon) | 1-NN | $k=1$, euclidean, uniform | 52.5 % | 0.523 | 53.6 % ($\pm 2.0\%$) |
| Sintético (mdlon) | K-NN default | $k=5$, euclidean, uniform | 55.3 % | 0.551 | 54.9 % ($\pm 1.8\%$) |
| Sintético (mdlon) | **K-NN mejor (CV train)** | **$k=31$, manhattan, uniform** | **62.0 %** | **0.611** | **61.1 % ($\pm 3.1\%$)** |
| Benchmark (WineQT) | 1-NN | $k=1$, euclidean, uniform | **66.2 %** | **0.644** | 63.6 % ($\pm 2.3\%$) |
| Benchmark (WineQT) | K-NN default | $k=5$, euclidean, uniform | 57.7 % | 0.535 | 58.7 % ($\pm 3.4\%$) |
| Benchmark (WineQT) | **K-NN mejor (CV train)** | **$k=31$, euclidean, distance** | 65.9 % | 0.632 | **68.2 % ($\pm 2.1\%$)** |

### Hallazgos de la exploración completa (36 configuraciones)

* **Sintético (mdlon):** La distancia **Manhattan** supera consistentemente a la Euclidiana en todos los valores de $k$ (alcanzando hasta 62.0 % en Hold-Out y 61.1 % en CV5 con $k=31$), mitigando la maldición de la dimensionalidad en los 500 rasgos. En este dataset los pesos `uniform` y `distance` producen exactamente las mismas predicciones debido a que las distancias entre puntos en alta dimensión tienden a ser equidistantes.
* **Benchmark (WineQT):** El ponderado por **distancia inversa (`distance`)** mejora drásticamente frente al voto uniforme (en $k=5$ pasa de 58.7 % a 65.3 % en CV5). Aunque $k=9$ (`euclidean`, `distance`) obtuvo el pico máximo en el conjunto Hold-Out puntual (70.6 %), la selección por CV en entrenamiento eligió $k=31$ (`euclidean`, `distance`), logrando la mejor estabilidad global en K-Fold (68.2 %).

## Matrices de confusión (Hold-Out 30%)

* **Sintético (mdlon) — K-NN mejor ($k=31$, manhattan, uniform):** Identifica con alta precisión la clase `-1` (231/300 aciertos, 77.0 % de *recall*), mientras que en la clase `1` acierta 141/300 (47.0 %).
* **Benchmark (WineQT) — K-NN mejor ($k=31$, euclidean, distance):** Buen desempeño en clase `0` (109/157 aciertos) y clase `1` (93/138 aciertos); la clase minoritaria `2` (calidad alta, 48 muestras) presenta mayor confusión hacia la clase media `1` (24 aciertos, 21 confundidos como clase `1`).

## Complejidad temporal y espacial

$n$ = muestras de entrenamiento, $m$ = muestras a clasificar, $d$ = rasgos, $k$ = vecinos evaluados.

| Clasificador | Entrenamiento (`fit`) | Predicción de $m$ muestras | Memoria |
|---|---|---|---|
| K-NN (desde cero) | $O(1)$ / $O(n \cdot d)$ por copia | $O(m \cdot n \cdot d + m \cdot n)$ | $O(n \cdot d)$ |

* **Mediciones empíricas de tiempo:**
  * `fit()` es prácticamente instantáneo y constante en ambos datasets ($\approx 0.02\text{–}0.05\text{ ms}$), ya que K-NN es un algoritmo perezoso (*lazy learner*) que solo almacena los arreglos en memoria.
  * Al escalar $n_{\text{train}}$ (del 25 % al 100 %), el tiempo total de predicción crece de forma casi lineal tanto en el Sintético (de $10.54\text{ ms}$ a $33.00\text{ ms}$ para $m=600$, pendiente log-log de $0.81$) como en WineQT (de $2.00\text{ ms}$ a $5.87\text{ ms}$ para $m=343$, pendiente log-log de $0.78$).
  * Al escalar $d$ (rasgos), el impacto en tiempo es sublineal gracias a la multiplicación matricial vectorizada de NumPy (`Xq @ X_train.T`), pasando de $23.26\text{ ms}$ ($d=125$) a $31.81\text{ ms}$ ($d=500$) en el sintético, y manteniéndose plano ($\approx 5.5\text{–}6.1\text{ ms}$) en WineQT.
  * La selección de los $k$ vecinos usa `np.argpartition` en $O(n)$ por consulta en lugar de ordenar toda la matriz de distancias en $O(n \log n)$.

## Comparativa entre los seis modelos

Exactitud promedio con Validación Cruzada ($k = 5$). *(Nota: A partir del Inciso 3, la práctica evalúa Wine agrupado en 3 clases, por lo que la columna de 6 clases marca N/A).*

| Modelo | Sintético | Wine (6 clases) | Wine (3 clases) | Responsable |
|---|---|---|---|---|
| Distancia Mínima | 59.9 % / 59.1 % | 23.8 % / 34.8 % | 36.6 % / 59.6 % | Inciso 2 |
| 1-NN | 63.4 % / 53.6 % | 56.0 % / 59.5 % | 59.5 % / 63.6 % | Inciso 2 |
| **K-NN (mejor CV)** | **61.1 %** | N/A | **68.2 %** | **Inciso 3** |
| Naïve Bayes | 58.0 % | N/A | 59.0 % | Inciso 4 |
| SVM | 60.0 % | N/A | 65.0 % | Inciso 4 |
| MLP | pendiente | N/A | pendiente | Inciso 5 |

## Conclusiones

1. **Ajustar hiperparámetros transforma el rendimiento de K-NN:** Con los valores por defecto ($k=5$, euclidiana, uniforme), K-NN obtiene resultados mediocres (54.9 % en Sintético y 58.7 % en WineQT), quedando incluso por debajo de 1-NN en WineQT. Al optimizar $k$, métrica y pesos mediante CV en entrenamiento, sube **+6.2 puntos** en el sintético (61.1 %) y **+9.5 puntos** en WineQT (68.2 %), superando también a Naïve Bayes y SVM.
2. **La métrica debe adaptarse a la dimensionalidad:** En espacios de alta dimensión con muchos rasgos ruidosos (500 en el sintético), la distancia **Manhattan** ($L_1$) es significativamente más robusta que la Euclidiana ($L_2$), elevando la exactitud en CV de 57.2 % a 61.1 % para $k=31$.
3. **Los pesos por distancia inversa (`distance`) evitan diluir la señal con $k$ grande:** En WineQT, usar $k=31$ con voto uniforme degrada el rendimiento en Hold-Out a 60.6 % porque vecinos lejanos de clases mayoritarias dominan la votación; al ponderar por $1/d$, se conserva la influencia de los vecinos más cercanos y la exactitud en CV alcanza 68.2 %.
4. **Vectorización y particionamiento parcial hacen viable K-NN desde cero:** Evitar ciclos por muestra mediante producto matricial y `np.argpartition` permite clasificar 600 muestras en 500 dimensiones en apenas **33 ms** totales ($0.055\text{ ms}$ por muestra).