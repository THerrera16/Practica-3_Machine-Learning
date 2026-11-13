# Inciso 2: Distancia Mínima y 1-NN (desde cero)

## Qué se hizo

- **Implementación desde cero** (solo NumPy) de **Distancia Mínima** (clasifica con el centroide más cercano de cada clase) y **1-NN** (clasifica con la clase del vecino de entrenamiento más cercano). Ambos usan distancia euclidiana. Sus resultados coinciden al 100 % con los de scikit-learn, que se usó solo para comprobarlos.
- **Datasets:** sintético (2000 muestras, 500 rasgos, 2 clases) y Wine (1143 muestras, 11 rasgos, calidad de 3 a 8). Como la práctica pide 3 clases, Wine se probó también agrupado en 3 (baja ≤5, media 6, alta ≥7).
- **Validación:** Hold Out 70/30 y K-Fold con k = 5, con las mismas particiones del resto del equipo.
- Cada modelo se probó **sin escalar** y con **z-score**.

## Resultados (exactitud)

| Dataset | Modelo | Hold Out (sin escalar / z-score) | K-Fold (sin escalar / z-score) |
|---|---|---|---|
| Sintético (mdlon) | Distancia Mínima | 59.2 % / 58.0 % | 59.9 % / 59.1 % |
| Sintético (mdlon) | 1-NN | 66.3 % / 52.5 % | 63.4 % / 53.6 % |
| Wine (6 clases) | Distancia Mínima | 23.6 % / 36.2 % | 23.8 % / 34.8 % |
| Wine (6 clases) | 1-NN | 54.5 % / 61.5 % | 56.0 % / 59.5 % |
| Wine (3 clases) | Distancia Mínima | 33.5 % / 58.6 % | 36.6 % / 59.6 % |
| Wine (3 clases) | 1-NN | 60.6 % / 66.2 % | 59.5 % / 63.6 % |

## Complejidad temporal

n = muestras de entrenamiento, m = muestras a clasificar, d = rasgos, c = clases.

| Clasificador | Entrenamiento | Predicción de m muestras | Memoria |
|---|---|---|---|
| Distancia Mínima | O(n·d) | O(m·c·d) | O(c·d) |
| 1-NN | O(n·d) al copiar los datos | O(m·n·d) | O(n·d) |

- **Distancia Mínima** solo guarda c centroides, por eso clasificar **no depende de n**.
- **1-NN** no entrena, pero cada muestra se compara contra las n de entrenamiento, así que todo el costo está en la predicción.
- Se comprobó midiendo tiempos: en 1-NN la predicción crece de forma lineal con n y con m (pendientes de 1.01 y 1.00 en escala log-log), y en Distancia Mínima no depende de n (0.00).

## Comparativa entre los seis modelos

Exactitud con K-Fold (k = 5). En Distancia Mínima y 1-NN se muestra "sin escalar / z-score". Las filas pendientes las llenan sus responsables con las mismas particiones.

| Modelo | Sintético | Wine (6 clases) | Wine (3 clases) | Responsable |
|---|---|---|---|---|
| Distancia Mínima | 59.9 % / 59.1 % | 23.8 % / 34.8 % | 36.6 % / 59.6 % | Inciso 2 |
| 1-NN | 63.4 % / 53.6 % | 56.0 % / 59.5 % | 59.5 % / 63.6 % | Inciso 2 |
| K-NN | pendiente | pendiente | pendiente | Inciso 3 |
| Naïve Bayes | pendiente | pendiente | pendiente | Inciso 4 |
| SVM | pendiente | pendiente | pendiente | Inciso 4 |
| MLP | pendiente | pendiente | pendiente | Inciso 5 |

## Conclusiones

1. **1-NN supera a Distancia Mínima en los tres datasets** (con el mejor escalado de cada uno), pero tarda entre unas 17 y 56 veces más en predecir y necesita guardar todo el entrenamiento.
2. **Distancia Mínima es débil cuando las clases se traslapan**, como en Wine, porque supone que cada clase es una nube compacta alrededor de su promedio.
3. **El escalado importa.** En Wine mejora mucho a Distancia Mínima (en 3 clases pasa de 36.6 % a 59.6 %) y unos 4 puntos a 1-NN, pero en el sintético empeora a 1-NN (de 63.4 % a 53.6 %), porque casi todos sus 500 rasgos son ruido y el z-score les da el mismo peso que a los útiles.
4. **Con 6 clases la exactitud engaña:** Distancia Mínima sin escalar saca 23.8 %, menos que predecir siempre la clase más frecuente (42.3 %).
