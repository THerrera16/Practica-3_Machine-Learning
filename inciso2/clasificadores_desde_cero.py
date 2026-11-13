"""
Inciso 2 - Clasificadores implementados DESDE CERO (solo NumPy):
  * DistanciaMinima  (clasificador de centroides más cercanos)
  * UnoNN            (1 vecino más cercano)

Ambos usan distancia euclidiana y comparten la interfaz:
    modelo.fit(X, y)  ->  entrena
    modelo.predict(X) ->  devuelve un arreglo con la clase predicha

No se usa ninguna clase de scikit-learn para clasificar.
"""
import numpy as np


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def distancia_cuadrada(A, B):
    """
    Matriz D de tamaño (m, n) con D[i, j] = ||A[i] - B[j]||^2.

    Se usa la identidad ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b, que es
    equivalente a la distancia euclidiana al cuadrado pero evita construir
    un tensor de tamaño (m, n, d) en memoria. Como la raíz cuadrada es
    monótona, comparar distancias al cuadrado da el mismo vecino más cercano.
    """
    a2 = np.sum(A * A, axis=1)[:, None]      # (m, 1)
    b2 = np.sum(B * B, axis=1)[None, :]      # (1, n)
    D = a2 + b2 - 2.0 * (A @ B.T)            # (m, n)
    np.maximum(D, 0.0, out=D)                # corrige errores de redondeo
    return D


def estandarizar(X_train, X_test):
    """Z-score con media y desviación calculadas SOLO con el entrenamiento
    (así no se filtra información del conjunto de prueba)."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0                  # evita dividir entre cero
    return (X_train - mu) / sigma, (X_test - mu) / sigma


# ----------------------------------------------------------------------
# Distancia Mínima (centroides)
# ----------------------------------------------------------------------
class DistanciaMinima:
    """Representa cada clase por el promedio (centroide) de sus muestras de
    entrenamiento y clasifica asignando la clase del centroide más cercano."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.clases_ = np.unique(y)
        self.centroides_ = np.array(
            [X[y == c].mean(axis=0) for c in self.clases_]
        )                                    # (c, d)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        D = distancia_cuadrada(X, self.centroides_)      # (m, c)
        return self.clases_[np.argmin(D, axis=1)]


# ----------------------------------------------------------------------
# 1-NN
# ----------------------------------------------------------------------
class UnoNN:
    """Memoriza el entrenamiento y clasifica cada muestra con la clase de su
    vecino más cercano. En caso de empate gana el primero en el orden del
    entrenamiento (np.argmin)."""

    def __init__(self, bloque=500):
        self.bloque = bloque                 # muestras de prueba por bloque

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.empty(len(X), dtype=self.y_.dtype)
        # Se procesa por bloques para que la matriz de distancias (bloque, n)
        # no crezca sin control cuando hay muchas muestras de prueba.
        for ini in range(0, len(X), self.bloque):
            fin = ini + self.bloque
            D = distancia_cuadrada(X[ini:fin], self.X_)
            pred[ini:fin] = self.y_[np.argmin(D, axis=1)]
        return pred


# ----------------------------------------------------------------------
# Métricas (también desde cero)
# ----------------------------------------------------------------------
def exactitud(y_real, y_pred):
    return float(np.mean(np.asarray(y_real) == np.asarray(y_pred)))


def matriz_confusion(y_real, y_pred, clases):
    """Filas = clase real, columnas = clase predicha."""
    idx = {c: i for i, c in enumerate(clases)}
    M = np.zeros((len(clases), len(clases)), dtype=int)
    for r, p in zip(np.asarray(y_real), np.asarray(y_pred)):
        M[idx[r], idx[p]] += 1
    return M


def f1_macro(y_real, y_pred, clases):
    """Promedio del F1 de cada clase (una clase sin muestras ni predicciones
    aporta 0). Es más informativo que la exactitud con clases desbalanceadas."""
    M = matriz_confusion(y_real, y_pred, clases)
    f1s = []
    for i in range(len(clases)):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))
