"""
K-NN implementado desde cero (solo NumPy).

Parámetros soportados (para las pruebas "distintos a los de default"):
  - k        : número de vecinos (k=1 equivale a 1-NN)
  - metric   : 'euclidean' | 'manhattan'
  - weights  : 'uniform' (voto simple) | 'distance' (voto ponderado por 1/d)


Los valores por default (los mismos que sklearn) son k=5, euclidean, uniform.

Empates en el voto: gana la clase con la etiqueta más pequeña (determinista).
"""
import numpy as np


class KNNDesdeCero:
    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric debe ser 'euclidean' o 'manhattan'")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights debe ser 'uniform' o 'distance'")
        self.k = k
        self.metric = metric
        self.weights = weights

    
    def fit(self, X, y):
        # K-NN es un 'lazy learner': fit solo memoriza los datos -> O(1)/O(n·d) por la copia.
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_, self.y_idx = np.unique(self.y_train, return_inverse=True)
        return self

    
    def _distancias(self, Xq):
        # Matriz de distancias (len(Xq), n_train).
        if self.metric == "euclidean":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b  (vectorizado, evita un ciclo por muestra)
            a2 = np.sum(Xq ** 2, axis=1)[:, None]
            b2 = np.sum(self.X_train ** 2, axis=1)[None, :]
            d2 = a2 + b2 - 2.0 * Xq @ self.X_train.T
            return np.sqrt(np.maximum(d2, 0.0))
        # manhattan: sum |a-b|, por bloques para no agotar memoria
        n_tr, d = self.X_train.shape
        presupuesto = 64 * 1024 * 1024 // 8  # ~64 MB de floats
        bloque = max(1, presupuesto // max(1, n_tr * d))
        out = np.empty((len(Xq), n_tr))
        for i in range(0, len(Xq), bloque):
            q = Xq[i:i + bloque]
            out[i:i + bloque] = np.abs(q[:, None, :] - self.X_train[None, :, :]).sum(axis=2)
        return out

    
    def kneighbors(self, X, k=None):
        # Devuelve (distancias, índices) de los k vecinos más cercanos, ordenados ascendente.
        k = self.k if k is None else k
        k = min(k, len(self.X_train))
        D = self._distancias(np.asarray(X, dtype=float))
        idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]           # O(n) por consulta
        d_k = np.take_along_axis(D, idx, axis=1)
        orden = np.argsort(d_k, axis=1, kind="stable")               # O(k log k)
        return np.take_along_axis(d_k, orden, axis=1), np.take_along_axis(idx, orden, axis=1)

    
    def votar(self, dist, idx, k=None, weights=None):
        # Predice a partir de vecinos ya calculados (permite probar varios k sin recalcular).
        k = self.k if k is None else k
        weights = self.weights if weights is None else weights
        dist, idx = dist[:, :k], idx[:, :k]
        etiquetas = self.y_idx[idx]                                  # (n, k) índices de clase
        if weights == "uniform":
            w = np.ones_like(dist)
        else:
            with np.errstate(divide="ignore"):
                w = 1.0 / dist
            exacto = np.isinf(w).any(axis=1)                         # coincidencia exacta (d=0)
            w[exacto] = (dist[exacto] == 0).astype(float)
        votos = np.stack([((etiquetas == c) * w).sum(axis=1)
                          for c in range(len(self.classes_))], axis=1)
        return self.classes_[np.argmax(votos, axis=1)]

    def predict(self, X):
        dist, idx = self.kneighbors(X, self.k)
        return self.votar(dist, idx, self.k, self.weights)
