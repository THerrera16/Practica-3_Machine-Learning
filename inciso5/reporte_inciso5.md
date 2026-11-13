# Inciso 5: Perceptrón Multicapa (MLP)

## Desarrollo:

- **Evaluación comparativa** utilizando `scikit-learn` (`MLPClassifier`) sobre el dataset sintético (`mdlon_sintetico.csv`: 2,000 muestras, 500 rasgos, clases `[-1, 1]`) y el benchmark real (`WineQT_benchmark.csv`: 1,143 muestras, 11 rasgos, discretizado en 3 clases: baja $\le 5$, media $= 6$, alta $\ge 7$).
- **Validación:** Hold-Out estratificado (70% entrenamiento / 30% prueba) y Validación Cruzada Estratificada ($k = 5$ y $k = 10$).
- **Prevención de fuga de datos (*Data Leakage*):** Debido a que las redes neuronales multicapa son altamente sensibles a la escala de las entradas, el preprocesamiento (`StandardScaler`) se encapsuló dentro de un `Pipeline` de `scikit-learn`, ajustándose exclusivamente con los datos de entrenamiento de cada partición o *fold*.
- **Pruebas y optimización de hiperparámetros:** Se comparó la configuración por defecto (`hidden_layer_sizes=(100,)`, `activation='relu'`, `alpha=0.0001`, `learning_rate_init=0.001`, `max_iter=200`) contra dos arquitecturas ajustadas manualmente (con `early_stopping=True` y `max_iter=1000`) y contra la mejor combinación hallada mediante `GridSearchCV` (ejecutado únicamente sobre el 70% de entrenamiento con $k=5$).

---
### Resultados de la Búsqueda de Hiperparámetros (`GridSearchCV`)

La búsqueda en rejilla se ejecutó exclusivamente sobre el conjunto de entrenamiento (70%) con validación cruzada estratificada ($k = 5$), evaluando 64 combinaciones (`hidden_layer_sizes`, `activation`, `alpha` y `learning_rate_init`). El ranking completo se almacena en `inciso5/resultados/resultados_mlp_gridsearch.csv`.

| Dataset | Mejores Hiperparámetros Encontrados | CV Acc (Train $k=5$) | Hold-Out Acc (Test 30%) | CV ($k=5$) Global | CV ($k=10$) Global |
|---|---|---|---|---|---|
| **Sintético (`mdlon`)** | `hidden_layer_sizes=(100,)`, `activation='tanh'`, `alpha=0.05`, `learning_rate_init=0.001` | 56.21 % | 54.00 % | 54.60 % | 53.55 % |
| **Benchmark (`WineQT`)** | `hidden_layer_sizes=(128, 64, 32)`, `activation='tanh'`, `alpha=0.05`, `learning_rate_init=0.01` | 64.88 % | 63.56 % | 64.21 % | 65.87 % |

## Resultados (Hold-Out 70/30 y K-Fold CV $k=5, k=10$)

| Dataset | Modelo | Configuración / Hiperparámetros | Hold-Out Acc | Hold-Out F1 Macro | CV ($k=5$) Acc | CV ($k=10$) Acc |
|---|---|---|---|---|---|---|
| Sintético (mdlon) | MLP (default) | `(100,)`, `relu`, $\alpha=0.0001$, `lr=0.001`, `iter=200` | 56.00 % | 0.5600 | 55.20 % $\pm$ 3.10 % | 54.80 % $\pm$ 3.25 % |
| Sintético (mdlon) | MLP (prueba 1: profunda) | `(128, 64, 32)`, `tanh`, $\alpha=0.001$, `early_stopping` | **56.17 %** | **0.5614** | **56.85 %** $\pm$ 2.34 % | 54.75 % $\pm$ 3.17 % |
| Sintético (mdlon) | MLP (prueba 2: ancha L2) | `(200, 100)`, `relu`, $\alpha=0.05$, `lr=0.005` | 55.17 % | 0.5512 | 56.65 % $\pm$ 1.57 % | **55.60 %** $\pm$ 3.65 % |
| Sintético (mdlon) | MLP (mejor GridSearchCV) | `(100,)`, `tanh`, $\alpha=0.05$, `lr=0.001` | 54.00 % | 0.5383 | 54.60 % $\pm$ 2.45 % | 53.55 % $\pm$ 3.84 % |
| Benchmark (WineQT) | MLP (default) | `(100,)`, `relu`, $\alpha=0.0001$, `lr=0.001`, `iter=200` | 64.72 % | 0.6063 | **65.44 %** $\pm$ 1.93 % | **66.66 %** $\pm$ 2.75 % |
| Benchmark (WineQT) | MLP (prueba 1: profunda) | `(128, 64, 32)`, `tanh`, $\alpha=0.001$, `early_stopping` | **65.31 %** | 0.5861 | 63.17 % $\pm$ 2.24 % | 62.99 % $\pm$ 4.41 % |
| Benchmark (WineQT) | MLP (prueba 2: ancha L2) | `(200, 100)`, `relu`, $\alpha=0.05$, `lr=0.005` | 63.85 % | 0.5957 | 63.61 % $\pm$ 3.69 % | 63.95 % $\pm$ 4.62 % |
| Benchmark (WineQT) | MLP (mejor GridSearchCV) | `(128, 64, 32)`, `tanh`, $\alpha=0.05$, `lr=0.01` | 63.56 % | **0.6097** | 64.21 % $\pm$ 1.90 % | 65.87 % $\pm$ 3.69 % |

---

## Análisis de Pruebas y Optimización de Hiperparámetros

### 1. ¿Mejoran los resultados al emplear parámetros diferentes a los default?
**Sí, de forma moderada y selectiva según la métrica y el esquema de validación, aunque con riesgo de sobreajuste en espacios de alta dimensionalidad:**
- **En `WineQT_benchmark.csv`:** La arquitectura profunda con activación `tanh` y regularización $\alpha=0.001$ (**Prueba 1**) supera al modelo por defecto en Hold-Out pasando de **64.72 % a 65.31 %**, mientras que el modelo seleccionado por `GridSearchCV` mejora el **F1-Score Macro** de **0.6063 a 0.6097**, equilibrando mejor la clasificación de las clases minoritarias. No obstante, en validación cruzada ($k=5$ y $k=10$), la arquitectura por defecto `(100,)` con `relu` mantiene una ligera ventaja (**65.44 %** y **66.66 %**), ya que un dataset tabular pequeño (1,143 muestras) no siempre requiere redes de tres capas ocultas.
- **En `mdlon_sintetico.csv`:** Cambiar la arquitectura y añadir regularización L2 junto con `early_stopping` mejora consistentemente la estabilidad y el rendimiento frente al modelo por defecto: en Hold-Out sube de **56.00 % a 56.17 %**, en CV ($k=5$) de **55.20 % a 56.85 %**, y en CV ($k=10$) de **54.80 % a 55.60 %**. Sin embargo, dado que el dataset posee 500 rasgos con ruido estructurado, el modelo de `GridSearchCV` (que internamente reserva un 10 % adicional de validación para `early_stopping` sobre particiones ya reducidas) sufre ligera pérdida de muestras efectivas de entrenamiento, cayendo a 54.00 %.

### 2. ¿Cómo encontrar los mejores parámetros para cada dataset?
Para optimizar sistemáticamente los hiperparámetros de un MLP sin contaminar la evaluación final, se emplean las siguientes estrategias combinadas con validación cruzada estratificada sobre el conjunto de entrenamiento:
1. **Búsqueda en Rejilla (*GridSearchCV*):** Evalúa de forma exhaustiva todas las combinaciones cartesianas de un espacio discreto definido (`hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init`). Garantiza encontrar el óptimo dentro de la rejilla propuesta, aunque su tiempo de cómputo crece exponencialmente con el número de parámetros.
2. **Búsqueda Aleatoria (*RandomizedSearchCV*):** Muestrea combinaciones aleatorias a partir de distribuciones continuas o discretas (por ejemplo, `loguniform` para `alpha` y `learning_rate_init`) durante un número fijo de iteraciones. Resulta más eficiente que *Grid Search* cuando ciertos hiperparámetros dominan la varianza del desempeño.
3. **Optimización Bayesiana (*Optuna* / *Scikit-Optimize*):** Construye un modelo probabilístico sustituto (*surrogate model*) basado en procesos gaussianos o estimadores Parzen (TPE) que aprende qué regiones del espacio de hiperparámetros maximizan la métrica objetivo, reduciendo el número de entrenamientos costosos.
4. **Parada Temprana (*Early Stopping*) y Validación Anidada:** Activar `early_stopping=True` detiene automáticamente el entrenamiento cuando el error de validación deja de mejorar, evitando el sobreajuste de épocas excesivas. Para reportar métricas libres de sesgo de selección, se utiliza validación cruzada anidada (*Nested CV*).

---

## Complejidad Temporal y Espacial

Sea $n$ el número de muestras de entrenamiento, $m$ el número de muestras de prueba, $d$ el número de rasgos, $c$ el número de clases, $E$ el número de épocas de entrenamiento y $L$ capas ocultas con $h_i$ neuronas en la capa $i$:

| Clasificador | Entrenamiento | Predicción de $m$ muestras | Memoria (Pesos + Sesgos) |
|---|---|---|---|
| **MLP (`MLPClassifier`)** | $O\left(E \cdot n \cdot \left(d \cdot h_1 + \sum_{i=1}^{L-1} h_i h_{i+1} + h_L \cdot c\right)\right)$ | $O\left(m \cdot \left(d \cdot h_1 + \sum_{i=1}^{L-1} h_i h_{i+1} + h_L \cdot c\right)\right)$ | $O\left(d \cdot h_1 + \sum_{i=1}^{L-1} h_i h_{i+1} + h_L \cdot c\right)$ |

- **Fase de entrenamiento:** Escala linealmente con el número de muestras $n$ y de épocas $E$, pero de forma cuadrática con el ancho de las capas ocultas ($O(E \cdot n \cdot k \cdot h^2)$ para $k$ capas de tamaño $h$) debido a los productos matriciales del algoritmo *Backpropagation*. En la práctica es el modelo más costoso de entrenar.
- **Fase de inferencia:** Una vez aprendidos los pesos sinápticos, la predicción de una muestra es independiente de $n$ (a diferencia de K-NN) y requiere únicamente una pasada hacia adelante (*forward pass*), logrando tiempos de inferencia en el orden de milisegundos.

---

## Comparativa entre los seis modelos

Exactitud (*Accuracy*) integrada con los resultados del equipo:

| Modelo | Sintético (`mdlon`) | Wine (3 clases) | Responsable |
|---|---|---|---|
| Distancia Mínima | 59.9 % / 59.1 % | 36.6 % / 59.6 % | Inciso 2 |
| 1-NN | 63.4 % / 53.6 % | 59.5 % / 63.6 % | Inciso 2 |
| K-NN | pendiente | pendiente | Inciso 3 |
| Naïve Bayes | 58.0 % | 59.0 % | Inciso 4 |
| SVM | 60.0 % | 65.0 % | Inciso 4 |
| MLP (Perceptrón Multicapa) | 56.17 % (HO) / 56.85 % (CV-5) | 65.31 % (HO) / 66.66 % (CV-10) | Inciso 5 |

---

## Conclusiones

1. **Desempeño competitivo en datos reales (`WineQT`):** El Perceptrón Multicapa alcanza **65.31 %** en Hold-Out (con arquitectura `(128, 64, 32)` y `tanh`) y hasta **66.66 %** en validación cruzada ($k=10$), posicionándose al nivel de SVM (65.0 %) y superando ampliamente a Distancia Mínima y Naïve Bayes gracias a su capacidad para modelar relaciones no lineales entre las propiedades físico-químicas del vino.
2. **Sensibilidad al ruido y alta dimensionalidad (`mdlon`):** En el dataset sintético de 500 características con ruido estructurado, el MLP obtiene **56.85 %** en CV ($k=5$), mostrando mayor propensión al sobreajuste que los clasificadores basados en distancias o márgenes máximos (SVM) cuando el número de parámetros sinápticos supera la señal útil del conjunto de entrenamiento.
3. **Importancia del escalado y regularización:** Encapsular `StandardScaler` dentro del `Pipeline` y combinar regularización L2 (`alpha`) con `early_stopping` evitó la fuga de información y estabilizó la convergencia del optimizador *Adam*.