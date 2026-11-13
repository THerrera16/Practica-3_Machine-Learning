# Inciso 4: Naïve Bayes y Máquinas de Vectores de Soporte (SVM)

## Desarrollo

* **Evaluación experimental** utilizando `scikit-learn` de **Gaussian Naïve Bayes** (`GaussianNB`) y **Support Vector Machines** (`SVC`) con búsqueda en malla (`GridSearchCV`) sobre dos conjuntos de datos:
  * **Sintético (`mdlon_sintetico.csv`):** 2000 muestras, 500 rasgos, 2 clases binarias ($-1, 1$).
  * **Benchmark (`WineQT_benchmark.csv`):** 1143 muestras, 11 rasgos físicos/químicos, agrupado en 3 clases de calidad: baja ($\le 5$, clase `0`), media ($= 6$, clase `1`) y alta ($\ge 7$, clase `2`).
* **Validación y mitigación de fuga de información (*Data Leakage*):**
  * **Hold-Out estratificado:** 70 % entrenamiento ($n_{\text{train}} = 1400$ en sintético; $n_{\text{train}} = 800$ en WineQT) y 30 % prueba ($n_{\text{test}} = 600$ en sintético; $n_{\text{test}} = 343$ en WineQT).
  * **K-Fold estratificado ($k = 5$):** Evaluado tanto en validación cruzada sobre el conjunto completo como dentro del proceso de `GridSearchCV`.
  * **Pipelines estandarizados:** El preprocesamiento de escalado (`StandardScaler`) se encapsuló dentro de un `Pipeline` de `scikit-learn` para garantizar que la media y varianza se calculen **exclusivamente dentro de los folds/muestras de entrenamiento** de cada iteración.
* **Búsqueda e hiperparámetros evaluados:**
  * **Naïve Bayes:** Se ajustó el parámetro de suavizado de varianza `var_smoothing` $\in \{10^{-12}, 10^{-9}, 10^{-6}, 10^{-3}, 10^{-2}, 10^{-1}\}$.
  * **SVM:** Se evaluó un espacio de búsqueda exhaustivo con kernels `linear`, `rbf`, `sigmoid` y `poly` (grados 2 y 3), variando $C \in \{0.1, 1, 10, 100\}$ y $\gamma \in \{\text{'scale'}, 0.001, 0.01, 0.1\}$.

---

## Resultados (Exactitud, F1-Score Macro y Validación Cruzada)

| Dataset | Modelo | Hiperparámetros / Configuración | Hold-Out Acc | Hold-Out F1 Macro | K-Fold ($k=5$) Acc | K-Fold ($k=5$) F1 Macro |
|---|---|---|---|---|---|---|
| Sintético (mdlon) | Naïve Bayes default | `var_smoothing`: 1e-09 (default) | 57.83 % | 0.5783 | 58.00 % ($\pm 1.62\%$) | 0.5799 |
| Sintético (mdlon) | Naïve Bayes ajustado | **`var_smoothing`: 1e-12** (elegido en Train) | 57.83 % | 0.5783 | 58.00 % ($\pm 1.62\%$) | 0.5799 |
| Sintético (mdlon) | SVM default | `kernel`: rbf, `C`: 1.0, `gamma`: scale | 58.83 % | 0.5883 | 59.80 % ($\pm 2.05\%$) | 0.5977 |
| Sintético (mdlon) | **SVM mejor (CV train)** | **`kernel`: rbf, `C`: 0.1, `gamma`: scale** | **60.00 %** | **0.5999** | **60.00 % ($\pm 1.48\%$)** | **0.5999** |
| Benchmark (WineQT) | Naïve Bayes default | `var_smoothing`: 1e-09 (default) | 58.89 % | 0.5629 | 59.05 % ($\pm 3.16\%$) | 0.5606 |
| Benchmark (WineQT) | Naïve Bayes ajustado | **`var_smoothing`: 1e-12** (elegido en Train) | 58.89 % | 0.5629 | 59.05 % ($\pm 3.16\%$) | 0.5606 |
| Benchmark (WineQT) | **SVM default** | **`kernel`: rbf, `C`: 1.0, `gamma`: scale** | **65.01 %** | **0.5878** | **64.92 % ($\pm 2.66\%$)** | **0.5843** |
| Benchmark (WineQT) | SVM mejor (CV train) | `kernel`: rbf, `C`: 100, `gamma`: 0.01 | 64.14 % | 0.5781 | 64.74 % ($\pm 1.77\%$) | 0.5880 |

### Hallazgos de la exploración completa (GridSearch)

1. **Sintético (mdlon):**
   * **Naïve Bayes:** El rendimiento fue insensible al parámetro `var_smoothing` (la exactitud se mantuvo idéntica en 58.0 % para todos los valores evaluados entre $10^{-12}$ y $10^{-1}$). Esto indica que la asunción de independencia condicional entre las 500 dimensiones ruidosas limita estructuralmente al modelo.
   * **SVM:** El mejor resultado en la mallas de entrenamiento fue alcanzado con un kernel `rbf` y regularización fuerte ($C = 0.1$, exactitud CV train = $60.00\%$), superando a los kernels `linear` (máximo 57.07 %) y `poly` (máximo 57.64 %). El margen blando restringido ($C$ pequeño) ayudó a evitar el sobreajuste sobre el ruido del dataset sintético.

2. **Benchmark (WineQT):**
   * **Naïve Bayes:** Obtiene 59.05 % en CV5. El clasificador sufre debido a las fuertes correlaciones entre variables físico-químicas (como acidez fija vs. ácido cítrico o densidad vs. alcohol), violando la suposición central del algoritmo.
   * **SVM:** Todos los kernels no lineales (`rbf`) dominaron la exploración. Durante el `GridSearchCV` en el 70 % de entrenamiento, la combinación `kernel: rbf, C: 100, gamma: 0.01` obtuvo el mayor puntaje interno (CV train = $66.50\%$). Sin embargo, al evaluar de manera independiente en la partición Hold-Out (30 %), el modelo con parámetros por defecto (`C: 1.0, gamma: 'scale'`) logró una exactitud ligeramente superior (**65.01 %** vs **64.14 %**), demostrando que valores elevados de $C$ introducen una leve varianza en la generalización.

---

## Matrices de confusión y métricas por clase (Hold-Out 30%)

### 1. Dataset Sintético (mdlon) — 600 muestras de prueba

* **Naïve Bayes (default / ajustado):** Exactitud: **57.83 %** | F1 Macro: **0.5783**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 171 & 129 \\ 124 & 176 \end{pmatrix}$$
  * **Desempeño:** Para la clase `-1`, *Precision* = 0.58, *Recall* = 0.57, *F1-score* = 0.57 (171/300 aciertos). Para la clase `1`, *Precision* = 0.58, *Recall* = 0.59, *F1-score* = 0.58 (176/300 aciertos). Presenta un comportamiento totalmente simétrico pero limitado por la alta dimensionalidad ruidosa.

* **SVM mejor (`rbf`, $C=0.1$, `gamma='scale'`):** Exactitud: **60.00 %** | F1 Macro: **0.5999**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 187 & 113 \\ 127 & 173 \end{pmatrix}$$
  * **Desempeño:** Muestra una mejora sustancial en la detección de la clase `-1` (*Recall* sube a **62.3 %**, 187/300 aciertos), manteniendo *Precision* de 0.60 en ambas clases.

---

### 2. Benchmark (WineQT) — 343 muestras de prueba

* **Naïve Bayes (default / ajustado):** Exactitud: **58.89 %** | F1 Macro: **0.5629**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 87 & 62 & 8 \\ 21 & 91 & 26 \\ 4 & 20 & 24 \end{pmatrix}$$
  * **Desempeño por clase:**
    * **Clase 0 (Baja $\le 5$, 157 muestras):** *Precision* = 0.78, *Recall* = 0.55, *F1* = 0.65. Sufre una elevada confusión clasificando 62 muestras de calidad baja como calidad media (`1`).
    * **Clase 1 (Media $= 6$, 138 muestras):** *Precision* = 0.53, *Recall* = 0.66, *F1* = 0.59.
    * **Clase 2 (Alta $\ge 7$, 48 muestras):** *Precision* = 0.41, *Recall* = 0.50, *F1* = 0.45 (24 aciertos, 20 confundidos con clase `1` y 4 con clase `0`).

* **SVM default (`rbf`, $C=1.0$, `gamma='scale'`):** Exactitud: **65.01 %** | F1 Macro: **0.5878**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 122 & 32 & 3 \\ 41 & 87 & 10 \\ 3 & 30 & 15 \end{pmatrix}$$
  * **Desempeño por clase:**
    * **Clase 0 (Baja):** *Precision* = 0.73, **Recall* = 77.7 %** (122/157 aciertos), *F1* = **0.76**. Reduce drásticamente los falsos positivos hacia la clase media.
    * **Clase 1 (Media):** *Precision* = 0.58, *Recall* = 63.0 % (87/138 aciertos), *F1* = 0.61.
    * **Clase 2 (Alta):** *Precision* = 0.54, *Recall* = 31.3 % (15/48 aciertos), *F1* = 0.39. Debido al desbalance de clases, SVM tiende a ser conservador con la clase minoritaria, favoreciendo la precisión global a costa del *recall* en los vinos de alta calidad.

* **SVM mejor GridSearchCV (`rbf`, $C=100$, $\gamma=0.01$):** Exactitud: **64.14 %** | F1 Macro: **0.5781**
  * **Matriz de confusión:**
    $$\begin{pmatrix} 118 & 35 & 4 \\ 40 & 84 & 14 \\ 3 & 28 & 17 \end{pmatrix}$$
  * Aumenta ligeramente los aciertos en la clase alta (`2`) a 17/48 (*Recall* = 35.4 %), pero degrada la precisión en las clases mayoritarias `0` y `1`.

---

## Complejidad temporal y espacial

$n$ = muestras de entrenamiento, $m$ = muestras a clasificar/evaluar, $d$ = rasgos/dimensiones, $c$ = clases, $S$ = número de vectores de soporte ($S \le n$).

| Clasificador | Entrenamiento (`fit`) | Predicción de $m$ muestras (`predict`) | Memoria |
|---|---|---|---|
| **Gaussian Naïve Bayes** | $O(n \cdot d)$ | $O(m \cdot c \cdot d)$ | $O(c \cdot d)$ |
| **SVM (SVC)** | $O(n^2 \cdot d)$ a $O(n^3 \cdot d)$ | $O(S \cdot d)$ | $O(S \cdot d + n \cdot d)$ |

### Mediciones empíricas de tiempo y escalabilidad

A partir del análisis de escalabilidad realizado variando la proporción del conjunto de entrenamiento ($n_{\text{train}}$ de 25 % a 100 %):

1. **Gaussian Naïve Bayes:**
   * **Entrenamiento (`fit`):** Extremadamente rápido y lineal. En el dataset sintético ($n=1400, d=500$) requiere apenas **$1.01\text{ ms}$**, mientras que en WineQT ($n=800, d=11$) tarda solo **$0.33\text{ ms}$**.
   * **Predicción (`predict`):** Prácticamente instantáneo: **$0.61\text{ ms}$** para clasificar 600 muestras en 500 dimensiones, y **$0.15\text{ ms}$** para 343 muestras en WineQT.

2. **Support Vector Machines (SVC):**
   * **Entrenamiento (`fit`):**
     * **Sintético ($d=500$):** Al pasar del 25 % ($n=350$) al 100 % ($n=1400$), el tiempo de ajuste de SVM default escala de **$15.22\text{ ms}$** a **$170.82\text{ ms}$**, reflejando la complejidad cuadrática respecto al número de muestras $n$.
     * **WineQT ($d=11$):** Escala de **$2.51\text{ ms}$** ($n=200$) a **$13.72\text{ ms}$** ($n=800$).
     * **Efecto de la optimización:** El modelo resultante de `GridSearchCV` ($C=100$) incrementa drásticamente el costo de ajuste debido a la menor tolerancia a violaciones del margen, requiriendo hasta **$446.88\text{ ms}$** en el dataset sintético.
   * **Predicción (`predict`):** Depende de la cantidad de vectores de soporte retenidos. En el dataset sintético, evaluar 600 muestras tomó **$85.83\text{ ms}$** (SVM default) y **$107.50\text{ ms}$** (SVM optimizado), lo cual es significativamente más lento que Naïve Bayes o K-NN debido al cálculo de las funciones de base radial ($K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$) contra los vectores de soporte.

---

## Comparativa general entre los seis modelos de la práctica

Exactitud promedio obtenida mediante **Validación Cruzada Estratificada ($k = 5$)**.

| Modelo | Sintético (`mdlon`) | Wine (6 clases) | Wine (3 clases) | Estado / Responsable |
|---|---|---|---|---|
| Distancia Mínima | 59.9 % / 59.1 % | 23.8 % / 34.8 % | 36.6 % / 59.6 % | Inciso 2 |
| 1-NN | 63.4 % / 53.6 % | 56.0 % / 59.5 % | 59.5 % / 63.6 % | Inciso 2 |
| K-NN (mejor CV) | 61.1 % | N/A | **68.2 %** | Inciso 3 |
| **Naïve Bayes** | **58.0 %** | N/A | **59.0 %** | **Inciso 4** |
| **SVM (mejor CV)** | **60.0 %** | N/A | **64.9 %** | **Inciso 4** |
| MLP (Red Neuronal) | pendiente | N/A | pendiente | Inciso 5 |

*(Nota: Las pruebas de Wine con 3 clases consolidan la evaluación homogénea del equipo a partir de los Incisos 3 y 4).*

---

## Conclusiones

1. **Las fronteras no lineales de SVM superan a los modelos probabilísticos simples:** En el dataset Benchmark (WineQT), SVM con kernel RBF alcanzó un **64.9 % en CV5 y 65.0 % en Hold-Out**, superando a Naïve Bayes por más de **6 puntos porcentuales** (59.0 %). La capacidad de proyectar los rasgos a un espacio de mayor dimensión mediante el kernel Gaussian permite separar de forma más efectiva las clases con distribución no lineal.
2. **Naïve Bayes se ve afectado por la correlación entre atributos:** En WineQT, la hipótesis de independencia condicional $P(x_1, x_2, \dots, x_d | y) = \prod P(x_i | y)$ no se cumple debido a la alta interdependencia entre variables químicas. Además, la sintonización de `var_smoothing` no generó impacto apreciable en la precisión global.
3. **Maldición de la dimensionalidad y ruido en el dataset sintético:** En el conjunto `mdlon` (500 dimensiones), tanto Naïve Bayes (58.0 %) como SVM (60.0 %) obtienen rendimientos modestos. Aunque SVM con regularización fuerte ($C=0.1$) logra filtrar parte del ruido, K-NN con distancia Manhattan (61.1 % en Inciso 3) logró una mejor adaptación al espacio hiperdimensional.
4. **Compromiso Eficiencia vs. Capacidad de Representación:** Gaussian Naïve Bayes es imbatible en términos de velocidad de inferencia ($0.61\text{ ms}$ para 600 muestras), lo que lo hace idóneo como modelo base (*baseline*) de bajísima complejidad. SVM ofrece mayor precisión pero con un costo computacional en $O(n^2 \cdot d)$ para entrenamiento y predicciones más lentas basadas en los vectores de soporte.
```
