# Inciso 4: Naïve Bayes y Máquinas de Vectores de Soporte (SVM)

## Qué se hizo

* **Evaluación experimental** utilizando `scikit-learn` de **Gaussian Naïve Bayes** (`GaussianNB`) y **Support Vector Machines** (`SVC`) con búsqueda en malla (`GridSearchCV`) sobre dos conjuntos de datos:
  * **Sintético (`mdlon_sintetico.csv`):** 2000 muestras, 500 rasgos, 2 clases binarias ($-1, 1$).
  * **Benchmark (`WineQT_benchmark.csv`):** 1143 muestras, 11 rasgos físicos/químicos, agrupado en 3 clases de calidad: baja ($\le 5$, clase `0`), media ($= 6$, clase `1`) y alta ($\ge 7$, clase `2`).
* **Validación y mitigación de fuga de información (*Data Leakage*):**
  * **Hold-Out estratificado:** 70 % entrenamiento ($n_{\text{train}} = 1400$ en sintético; $n_{\text{train}} = 800$ en WineQT) y 30 % prueba ($n_{\text{test}} = 600$ en sintético; $n_{\text{test}} = 343$ en WineQT).
  * **K-Fold estratificado ($k = 5$):** Evaluado tanto en validación cruzada sobre el conjunto completo como dentro del proceso de `GridSearchCV`.
  * **Pipelines estandarizados:** El preprocesamiento de escalado (`StandardScaler`) se encapsuló dentro de un `Pipeline` de `scikit-learn` para garantizar que la media y varianza se calculen **exclusivamente dentro de los folds/muestras de entrenamiento** de cada iteración.
* **Búsqueda e hiperparámetros evaluados:**
  * **Naïve Bayes:** Se ajustó el parámetro de suavizado de varianza `var_smoothing` $\in \{10^{-12}, 10^{-9}, 10^{-6}, 10^{-3}, 10^{-2}, 10^{-1}\}$.
  * **SVM (28 configuraciones por dataset):** kernel `rbf` con $C \in \{0.1, 1, 10, 100\}$ y $\gamma \in \{\text{'scale'}, 0.001, 0.01, 0.1\}$ (16 combinaciones); kernels `linear`, `sigmoid` y `poly` (grados 2 y 3, $\gamma=\text{'scale'}$) con $C \in \{0.1, 1, 10\}$.

---

## Resultados (Exactitud, F1-Score Macro y Validación Cruzada)

| Dataset | Modelo | Hiperparámetros / Configuración | Hold-Out Acc | Hold-Out F1 Macro | K-Fold ($k=5$) Acc | K-Fold ($k=5$) F1 Macro |
|---|---|---|---|---|---|---|
| Sintético (mdlon) | Naïve Bayes default | `var_smoothing`: 1e-09 (default) | 57.83 % | 0.5783 | **59.65 % ($\pm 3.01\%$)** | **0.5961** |
| Sintético (mdlon) | Naïve Bayes ajustado | `var_smoothing`: 1e-12 (elegido en Train) | 57.83 % | 0.5783 | **59.65 % ($\pm 3.01\%$)** | **0.5961** |
| Sintético (mdlon) | SVM default | `kernel`: rbf, `C`: 1.0, `gamma`: scale | 58.83 % | 0.5883 | 57.70 % ($\pm 2.84\%$) | 0.5764 |
| Sintético (mdlon) | **SVM mejor (CV train)** | `kernel`: rbf, `C`: 0.1, `gamma`: scale | **60.00 %** | **0.5998** | 59.30 % ($\pm 3.09\%$) | 0.5908 |
| Benchmark (WineQT) | Naïve Bayes default | `var_smoothing`: 1e-09 (default) | 58.89 % | 0.5616 | 60.37 % ($\pm 2.95\%$) | 0.5862 |
| Benchmark (WineQT) | Naïve Bayes ajustado | `var_smoothing`: 1e-12 (elegido en Train) | 58.89 % | 0.5616 | 60.37 % ($\pm 2.95\%$) | 0.5862 |
| Benchmark (WineQT) | **SVM default** | `kernel`: rbf, `C`: 1.0, `gamma`: scale | **65.31 %** | **0.5855** | **65.79 % ($\pm 1.33\%$)** | **0.6163** |
| Benchmark (WineQT) | SVM mejor (CV train) | `kernel`: rbf, `C`: 100, `gamma`: 0.01 | 63.85 % | 0.5804 | 65.36 % ($\pm 2.95\%$) | 0.6139 |

*(En negrita, el mejor valor de cada columna dentro de cada dataset.)*

### Hallazgos de la exploración completa (GridSearch)

1. **Sintético (mdlon):**
   * **Naïve Bayes:** La exactitud no cambió al pasar de `var_smoothing` $=10^{-9}$ (default) a $10^{-12}$ (ajustado). Esto indica que la asunción de independencia condicional entre las 500 dimensiones ruidosas limita estructuralmente al modelo, no el suavizado.
   * **SVM:** El mejor resultado en la malla (CV dentro del entrenamiento) fue con kernel `rbf` y regularización fuerte ($C = 0.1$, `gamma='scale'`, exactitud CV train = $59.21\%$). Los demás kernels quedaron por debajo: `poly` (máximo 58.57 %), `sigmoid` (máximo 58.00 %) y `linear` (máximo 52.50 %). El margen blando amplio ($C$ pequeño) ayudó a evitar el sobreajuste sobre el ruido del dataset sintético. Además, el kernel `linear` fue extremadamente costoso en 500 dimensiones (tiempo medio de ajuste de 4.3 s con $C=0.1$, 38.2 s con $C=1$ y 299.2 s con $C=10$, frente a 0.6–1.7 s del `rbf`).
   * En CV5 sobre el conjunto completo, Naïve Bayes (59.65 %) y SVM mejor (59.30 %) son estadísticamente indistinguibles (desviaciones de ≈ 3 puntos).

2. **Benchmark (WineQT):**
   * **Naïve Bayes:** Obtiene 60.37 % en CV5. El clasificador sufre debido a las fuertes correlaciones entre variables físico-químicas (como acidez fija vs. ácido cítrico o densidad vs. alcohol), violando la suposición central del algoritmo.
   * **SVM:** El kernel `rbf` dominó la exploración (los cinco primeros lugares son `rbf`). Durante el `GridSearchCV` en el 70 % de entrenamiento, la combinación `kernel: rbf, C: 100, gamma: 0.01` obtuvo el mayor puntaje interno (CV train = $65.00\%$), seguida de `C: 10, gamma: 'scale'` (64.88 %). Los kernels `poly` (máximo 63.00 %) y `linear` (máximo 62.88 %) quedaron cerca, y `sigmoid` muy por debajo (máximo 57.88 %). Sin embargo, al evaluar de manera independiente, el modelo con parámetros por defecto (`C: 1.0, gamma: 'scale'`) resultó ligeramente superior tanto en Hold-Out (**65.31 %** vs **63.85 %**) como en CV5 completo (65.79 % vs 65.36 %). La diferencia es pequeña frente a la desviación estándar (1.3–3 puntos), por lo que no hay evidencia clara de que ajustar $C$ y $\gamma$ mejore la generalización; valores elevados de $C$ no aportaron beneficio.

---

## Matrices de confusión y métricas por clase (Hold-Out 30%)

### 1. Dataset Sintético (mdlon) — 600 muestras de prueba

* **Naïve Bayes (default / ajustado):** Exactitud: **57.83 %** | F1 Macro: **0.5783**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 171 & 129 \\ 124 & 176 \end{pmatrix}$$
  * **Desempeño:** Para la clase `-1`, *Precision* = 0.58, *Recall* = 0.57, *F1-score* = 0.57 (171/300 aciertos). Para la clase `1`, *Precision* = 0.58, *Recall* = 0.59, *F1-score* = 0.58 (176/300 aciertos). Presenta un comportamiento simétrico pero limitado por la alta dimensionalidad ruidosa.

* **SVM mejor (`rbf`, $C=0.1$, `gamma='scale'`):** Exactitud: **60.00 %** | F1 Macro: **0.5998**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 187 & 113 \\ 127 & 173 \end{pmatrix}$$
  * **Desempeño:** Mejora la detección de la clase `-1` (*Recall* sube a **62.3 %**, 187/300 aciertos, frente a 57.0 % de Naïve Bayes), manteniendo *Precision* de 0.60 en ambas clases.

---

### 2. Benchmark (WineQT) — 343 muestras de prueba

* **Naïve Bayes (default / ajustado):** Exactitud: **58.89 %** | F1 Macro: **0.5616**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 87 & 62 & 8 \\ 21 & 91 & 26 \\ 4 & 20 & 24 \end{pmatrix}$$
  * **Desempeño por clase:**
    * **Clase 0 (Baja $\le 5$, 157 muestras):** *Precision* = 0.78, *Recall* = 0.55, *F1* = 0.65. Sufre una elevada confusión clasificando 62 muestras de calidad baja como calidad media (`1`).
    * **Clase 1 (Media $= 6$, 138 muestras):** *Precision* = 0.53, *Recall* = 0.66, *F1* = 0.59.
    * **Clase 2 (Alta $\ge 7$, 48 muestras):** *Precision* = 0.41, *Recall* = 0.50, *F1* = 0.45 (24 aciertos, 20 confundidos con clase `1` y 4 con clase `0`).

* **SVM default (`rbf`, $C=1.0$, `gamma='scale'`):** Exactitud: **65.31 %** | F1 Macro: **0.5855**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 122 & 32 & 3 \\ 41 & 87 & 10 \\ 3 & 30 & 15 \end{pmatrix}$$
  * **Desempeño por clase:**
    * **Clase 0 (Baja):** *Precision* = 0.73, *Recall* = **77.7 %** (122/157 aciertos), *F1* = **0.76**. Reduce a la mitad las muestras de calidad baja clasificadas como media (32 frente a 62 de Naïve Bayes).
    * **Clase 1 (Media):** *Precision* = 0.58, *Recall* = 63.0 % (87/138 aciertos), *F1* = 0.61.
    * **Clase 2 (Alta):** *Precision* = 0.54, *Recall* = 31.3 % (15/48 aciertos), *F1* = 0.39. Debido al desbalance de clases, SVM tiende a ser conservador con la clase minoritaria, favoreciendo la exactitud global a costa del *recall* en los vinos de alta calidad.

* **SVM mejor GridSearchCV (`rbf`, $C=100$, $\gamma=0.01$):** Exactitud: **63.85 %** | F1 Macro: **0.5804**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 118 & 35 & 4 \\ 40 & 84 & 14 \\ 3 & 28 & 17 \end{pmatrix}$$
  * Aumenta ligeramente los aciertos en la clase alta (`2`) a 17/48 (*Recall* = 35.4 %), pero reduce un poco el *recall* de las clases mayoritarias `0` (75.2 %) y `1` (60.9 %).

---

## Complejidad temporal y espacial

$n$ = muestras de entrenamiento, $m$ = muestras a clasificar/evaluar, $d$ = rasgos/dimensiones, $c$ = clases, $S$ = número de vectores de soporte ($S \le n$).

| Clasificador | Entrenamiento (`fit`) | Predicción de $m$ muestras (`predict`) | Memoria |
|---|---|---|---|
| **Gaussian Naïve Bayes** | $O(n \cdot d)$ | $O(m \cdot c \cdot d)$ | $O(c \cdot d)$ |
| **SVM (SVC)** | $O(n^2 \cdot d)$ a $O(n^3 \cdot d)$ | $O(m \cdot S \cdot d)$ | $O(S \cdot d + n \cdot d)$ |

### Mediciones empíricas de tiempo y escalabilidad

A partir del análisis de escalabilidad variando la proporción del conjunto de entrenamiento ($n_{\text{train}}$ de 25 % a 100 %). Son mediciones de una sola corrida, por lo que tienen ruido (sobre todo en tiempos de pocos milisegundos).

1. **Gaussian Naïve Bayes:**
   * **Entrenamiento (`fit`):** En el dataset sintético crece de forma aproximadamente lineal con $n$ (de $9.09\text{ ms}$ con $n=350$ a $37.53\text{ ms}$ con $n=1400$, ×4.1 al cuadruplicar $n$). En WineQT ($d=11$) se mantiene casi constante ($\approx 3.1\text{–}3.3\text{ ms}$), dominado por el costo fijo.
   * **Predicción (`predict`):** No depende de $n$. Tarda entre $7.5$ y $13.4\text{ ms}$ para clasificar 600 muestras en 500 dimensiones, y $\approx 0.82\text{ ms}$ para 343 muestras en WineQT.

2. **Support Vector Machines (SVC):**
   * **Entrenamiento (`fit`):**
     * **Sintético ($d=500$):** Al pasar del 25 % ($n=350$) al 100 % ($n=1400$), el tiempo de ajuste de SVM default escala de **$43.56\text{ ms}$** a **$364.98\text{ ms}$** (×8.4 al cuadruplicar $n$, exponente empírico $\approx 1.5$): crecimiento claramente superlineal, aunque por debajo del cuadrático puro.
     * **WineQT ($d=11$):** Escala de **$6.54\text{ ms}$** ($n=200$) a **$17.79\text{ ms}$** ($n=800$); con tan pocas muestras el costo fijo domina y no se aprecia el crecimiento cuadrático.
     * **Efecto de la optimización:** En WineQT el modelo de `GridSearchCV` ($C=100$, $\gamma=0.01$) tarda **$31.25\text{ ms}$** frente a $17.79\text{ ms}$ del default (≈ ×1.8), por la menor tolerancia a violaciones del margen. En el sintético el modelo elegido ($C=0.1$) no es más costoso que el default ($308.63\text{ ms}$ vs $364.98\text{ ms}$).
   * **Predicción (`predict`):** Depende de la cantidad de vectores de soporte retenidos. En el dataset sintético, evaluar 600 muestras tomó **$279.26\text{ ms}$** (SVM default) y **$275.04\text{ ms}$** (SVM optimizado); en WineQT, $\approx 19\text{ ms}$ para 343 muestras en ambos casos. Esto es entre ≈ 9 y 21 veces más lento que K-NN y Naïve Bayes en el sintético (33 ms y 13 ms respectivamente), debido al cálculo de las funciones de base radial ($K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$) contra los vectores de soporte.

---

## Comparativa general entre los seis modelos de la práctica

Exactitud promedio obtenida mediante **Validación Cruzada Estratificada ($k = 5$)**. En Distancia Mínima y 1-NN se muestra "sin escalar / z-score"; K-NN, Naïve Bayes y SVM se evaluaron con z-score.

| Modelo | Sintético (`mdlon`) | Wine (6 clases) | Wine (3 clases) | Estado / Responsable |
|---|---|---|---|---|
| Distancia Mínima | 59.9 % / 59.1 % | 23.8 % / 34.8 % | 36.6 % / 59.6 % | Inciso 2 |
| 1-NN | 63.4 % / 53.6 % | 56.0 % / 59.5 % | 59.5 % / 63.6 % | Inciso 2 |
| K-NN (mejor CV) | 61.1 % | N/A | **68.2 %** | Inciso 3 |
| **Naïve Bayes** | **59.7 %** | N/A | **60.4 %** | **Inciso 4** |
| **SVM (RBF, mejor GridSearchCV)** | **59.3 %** | N/A | **65.4 %** | **Inciso 4** |
| MLP (Red Neuronal) | pendiente | N/A | pendiente | Inciso 5 |

*(Nota: Las pruebas de Wine con 3 clases consolidan la evaluación homogénea del equipo a partir de los Incisos 3 y 4. El SVM por defecto obtiene 65.8 % en Wine con 3 clases y 57.7 % en el sintético.)*

---

## Conclusiones

1. **Las fronteras no lineales de SVM superan a Naïve Bayes en Wine:** En el dataset Benchmark (WineQT), SVM con kernel RBF alcanzó **65.4–65.8 % en CV5** y **63.9–65.3 % en Hold-Out**, frente a **60.4 %** y **58.9 %** de Naïve Bayes, es decir, una ventaja de ≈ 5 puntos en CV5 y de ≈ 5–6 puntos en Hold-Out. La capacidad de proyectar los rasgos a un espacio de mayor dimensión mediante el kernel Gaussiano permite separar mejor clases con distribución no lineal.
2. **Naïve Bayes se ve afectado por la correlación entre atributos:** En WineQT, la hipótesis de independencia condicional $P(x_1, x_2, \dots, x_d | y) = \prod P(x_i | y)$ no se cumple debido a la alta interdependencia entre variables químicas. Además, cambiar `var_smoothing` de $10^{-9}$ a $10^{-12}$ no generó ningún cambio en la exactitud.
3. **Maldición de la dimensionalidad y ruido en el dataset sintético:** En el conjunto `mdlon` (500 dimensiones), Naïve Bayes (59.7 % en CV5) y SVM (59.3 %) obtienen rendimientos modestos y prácticamente equivalentes. Aunque SVM con regularización fuerte ($C=0.1$) logra el mejor Hold-Out de los dos (60.0 %), K-NN con distancia Manhattan (61.1 % en Inciso 3) logró una mejor adaptación al espacio hiperdimensional.
4. **Compromiso Eficiencia vs. Capacidad de Representación:** Gaussian Naïve Bayes es el más rápido en inferencia ($\approx 13\text{ ms}$ para 600 muestras en el sintético y $0.8\text{ ms}$ para 343 en Wine), lo que lo hace idóneo como modelo base (*baseline*). SVM ofrece mayor exactitud en Wine pero con un costo de entrenamiento superlineal en $n$ y predicciones más lentas basadas en los vectores de soporte; además, afinar sus hiperparámetros no mejoró la generalización respecto a los valores por defecto.
