# Inciso 4: Naïve Bayes y Máquinas de Vectores de Soporte (SVM)

## Desarrollo:

- **Evaluación comparativa** utilizando `scikit-learn` de **Gaussian Naïve Bayes** (`GaussianNB`) y **Support Vector Machines** (`SVC`) sobre el dataset sintético (`mdlon_sintetico.csv`) y el benchmark real (`WineQT_benchmark.csv` con 3 clases: baja $\le 5$, media $= 6$, alta $\ge 7$).
- **Validación:** Hold-Out estratificado (70% entrenamiento / 30% prueba) y Validación Cruzada Estratificada ($k = 5$).
- **Prevención de fuga de datos (*Data Leakage*):** El preprocesamiento de estandarización (`StandardScaler`) se encapsuló dentro de un `Pipeline` de `scikit-learn`, ajustándose exclusivamente sobre los datos de entrenamiento de cada partición o *fold*.
- **Optimización de hiperparámetros:** Búsqueda en malla (`GridSearchCV`) ejecutada únicamente sobre el 70% de entrenamiento (con $k=5$ y optimizando por *accuracy*). Para Naïve Bayes se ajustó `var_smoothing`, y para SVM se evaluaron los kernels `linear`, `rbf`, `sigmoid` y `poly` variando $C$, $\gamma$ y el grado del polinomio.

## Resultados (Exactitud / Hold-Out)

| Dataset | Modelo | Configuración / Hiperparámetros | Hold-Out Acc | F1-Score Macro |
|---|---|---|---|---|
| Sintético (mdlon) | Naive Bayes | Default | 58.0 % | 0.58 |
| Sintético (mdlon) | Naive Bayes | `var_smoothing`: 1e-12 | 58.0 % | 0.58 |
| Sintético (mdlon) | SVM | Default (`rbf`, C=1.0) | 59.0 % | 0.59 |
| Sintético (mdlon) | SVM | GridSearchCV (`rbf`, C=0.1) | **60.0 %** | 0.60 |
| Benchmark (WineQT) | Naive Bayes | Default | 59.0 % | 0.56 |
| Benchmark (WineQT) | Naive Bayes | `var_smoothing`: 1e-12 | 59.0 % | 0.56 |
| Benchmark (WineQT) | SVM | Default (`rbf`, C=1.0) | **65.0 %** | 0.59 |
| Benchmark (WineQT) | SVM | GridSearchCV (`rbf`, C=100, $\gamma$=0.01) | 64.0 % | 0.58 |

## Complejidad temporal y espacial

n = muestras de entrenamiento, m = muestras de prueba, d = rasgos, c = clases, S = número de vectores de soporte.

| Clasificador | Entrenamiento | Predicción de m muestras | Memoria |
|---|---|---|---|
| Gaussian Naïve Bayes | $O(n \cdot d)$ | O(m \cdot c \cdot d) | $O(c \cdot d)$ |
| SVM (SVC) | $O(n^2 \cdot d)$ a $O(n^3 \cdot d)$ | $O(S \cdot d)$ | $O(S \cdot d)$ |

- **Naïve Bayes** destaca por su extrema rapidez de entrenamiento y predicción al asumir independencia condicional entre rasgos.
- **SVM** escala de manera cuadrática o cúbica con respecto a las muestras de entrenamiento en su etapa de optimización, pero su tiempo de inferencia depende únicamente del número de vectores de soporte seleccionados.

## Comparativa entre los seis modelos

Exactitud con Validación Cruzada ($k = 5$) o evaluación general integrada con el equipo.

| Modelo | Sintético | Wine (6 clases) | Wine (3 clases) | Responsable |
|---|---|---|---|---|
| Distancia Mínima | 59.9 % / 59.1 % | 23.8 % / 34.8 % | 36.6 % / 59.6 % | Inciso 2 |
| 1-NN | 63.4 % / 53.6 % | 56.0 % / 59.5 % | 59.5 % / 63.6 % | Inciso 2 |
| K-NN | pendiente | pendiente | pendiente | Inciso 3 |
| Naïve Bayes | 58.0 % | pendiente | 59.0 % | Inciso 4 |
| SVM | 60.0 % | pendiente | 65.0 % | Inciso 4 |
| MLP | pendiente | pendiente | pendiente | Inciso 5 |

## Conclusiones

1. **Superioridad de SVM en datos complejos:** En el dataset WineQT, SVM supera claramente a Naïve Bayes (65.0% frente a 59.0%), capturando mejor las fronteras de decisión no lineales gracias al kernel RBF.
2. **Impacto marginal de la optimización en Naïve Bayes:** Ajustar el parámetro `var_smoothing` mediante `GridSearchCV` no aportó mejoras de rendimiento perceptibles frente a los valores por defecto en ninguno de los dos datasets.
3. **Complejidad del dataset sintético:** Ambos modelos obtienen rendimientos cercanos al límite inferior (~58-60%) en el dataset sintético `mdlon`, lo que sugiere un alto nivel de solapamiento o ruido intrínseco en sus 500 características.
4. **Prevención de sobreajuste:** El uso estricto de `Pipeline` con `StandardScaler` y validación cruzada garantizó que no existiera contaminación de información (*data leakage*) durante la búsqueda de hiperparámetros de SVM.
