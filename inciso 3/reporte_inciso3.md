# Inciso 3: K-Nearest Neighbors (K-NN desde cero)

## Desarrollo

- **Implementación desde cero** (solo NumPy) de **K-NN** (`KNNDesdeCero`) soportando múltiples valores de $k$, distancias euclidiana y Manhattan, y esquemas de peso uniforme o ponderado por distancia ($1/d$). Los resultados coinciden con los de `scikit-learn`, usado exclusivamente como validación.
- **Datasets:** Sintético (`mdlon_sintetico.csv`) y Benchmark (`WineQT_benchmark.csv` evaluado en 3 clases: baja $\le 5$, media $= 6$, alta $\ge 7$).
- **Validación y Optimización:** Hold-Out estratificado (70% entrenamiento / 30% prueba) y Validación Cruzada Estratificada ($k = 5$). La selección de hiperparámetros óptimos ($k$, métrica y pesos) se realizó mediante CV **estrictamente sobre el 70% de entrenamiento**, manteniendo el test completamente aislado.
- **Prevención de fuga de datos (*Data Leakage*):** El preprocesamiento de estandarización (`StandardScaler`) se ajustó de forma independiente en cada partición o *fold* de entrenamiento.

## Resultados (Exactitud y Métricas)

| Dataset | Modelo | Hiperparámetros ($k$, métrica, pesos) | Hold-Out Acc | F1-Score Macro | CV (k=5) Acc media |
|---|---|---|---|---|---|
| Sintético (mdlon) | 1-NN | $k=1$, euclidiana, uniform | 52.5 % | 0.523 | 53.6 % |
| Sintético (mdlon) | K-NN default | $k=5$, euclidiana, uniform | 55.3 % | 0.551 | 54.9 % |
| Sintético (mdlon) | K-NN mejor | $k=31$, manhattan, uniform | **62.0 %** | 0.611 | 61.1 % |
| Benchmark (WineQT) | 1-NN | $k=1$, euclidiana, uniform | 66.2 % | 0.644 | 63.6 % |
| Benchmark (WineQT) | K-NN default | $k=5$, euclidiana, uniform | 57.7 % | 0.535 | 58.7 % |
| Benchmark (WineQT) | K-NN mejor | $k=31$, euclidiana, distance | 65.9 % | 0.632 | **68.2 %** |

## Complejidad temporal y espacial

n = muestras de entrenamiento, m = muestras a clasificar, d = rasgos, k = vecinos evaluados.

| Fase / Operación | Costo Temporal | Costo Espacial |
|---|---|---|
| Entrenamiento (`fit`) | $O(1)$ (aprendizaje perezoso / copia de referencias) | $O(n \cdot d)$ |
| Cálculo de distancias | $O(m \cdot n \cdot d)$ | $O(m \cdot n)$ |
| Inferencia (`predict`) | $O(m \cdot n \cdot d + m \cdot k \log k)$ | $O(n \cdot d)$ |

- **Costo de Inferencia:** Al ser un clasificador *lazy* (sin entrenamiento real), todo el costo computacional recae sobre la etapa de predicción al calcular distancias contra las $n$ muestras de entrenamiento.
- **Optimización interna:** La implementación vectorizada calcula eficientemente las matrices de distancia y utiliza particionamiento parcial (`np.argpartition`) para evitar ordenar la totalidad de los datos de entrenamiento por cada consulta.

## Comparativa entre los seis modelos

Exactitud con Validación Cruzada ($k = 5$). 

| Modelo | Sintético | Wine (6 clases) | Wine (3 clases) | Responsable |
|---|---|---|---|---|
| Distancia Mínima | 59.9 % / 59.1 % | 23.8 % / 34.8 % | 36.6 % / 59.6 % | Inciso 2 |
| 1-NN | 63.4 % / 53.6 % | 56.0 % / 59.5 % | 59.5 % / 63.6 % | Inciso 2 |
| K-NN (Mejor) | **61.1 %** | pendiente | **68.2 %** | Inciso 3 |
| Naïve Bayes | 58.0 % | pendiente | 59.0 % | Inciso 4 |
| SVM | 60.0 % | pendiente | 65.0 % | Inciso 4 |
| MLP | pendiente | pendiente | pendiente | Inciso 5 |

## Conclusiones

1. **Beneficio de un $k$ mayor:** En ambos datasets, ampliar el vecindario a $k=31$ mejoró sustancialmente el rendimiento frente a la configuración por defecto ($k=5$), suavizando las fronteras de decisión frente al ruido local.
2. **Influencia de la métrica y los pesos:** En el dataset sintético, la métrica Manhattan con votos uniformes maximizó la exactitud (62%), mientras que en WineQT la métrica Euclidiana combinada con pesos ponderados por distancia (`distance`) alcanzó el mejor puntaje global de validación cruzada (68.2%).
3. **Robustez frente a Distancia Mínima:** K-NN supera ampliamente a los clasificadores basados en centroides únicos (Distancia Mínima), ya que es capaz de modelar fronteras no lineales y distribuciones multimodales en los datos.
4. **Costo computacional:** Si bien la fase de entrenamiento es instantánea, la predicción de conjuntos grandes se vuelve costosa debido a la naturaleza de aprendizaje perezoso del algoritmo.
