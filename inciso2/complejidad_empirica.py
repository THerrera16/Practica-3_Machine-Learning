"""
Evidencia empírica para el análisis de complejidad temporal.

Se mide el tiempo de predicción al variar UNA variable a la vez y se estima
la pendiente en escala log-log (pendiente ~ 1  =>  crecimiento lineal).
Los datos son aleatorios porque el tiempo solo depende de los tamaños.

  n = muestras de entrenamiento, m = muestras de prueba,
  d = número de rasgos,          c = número de clases
"""
import time

import numpy as np
from threadpoolctl import threadpool_limits   # viene instalado con scikit-learn

from clasificadores_desde_cero import DistanciaMinima, UnoNN

# NumPy reparte la multiplicación de matrices entre varios hilos; eso agrega
# ruido a los tiempos. Con 1 hilo las mediciones son mucho más estables.
threadpool_limits(limits=1)

rng = np.random.default_rng(0)


def tiempo(f, repeticiones=15):
    """Mejor tiempo de varias repeticiones (reduce el ruido del sistema).
    La primera llamada es de calentamiento y no se cuenta."""
    f()
    mejores = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        f()
        mejores.append(time.perf_counter() - t0)
    return min(mejores)


def pendiente_loglog(xs, ts):
    return np.polyfit(np.log(xs), np.log(ts), 1)[0]


def datos(n, d, c, m):
    Xtr = rng.normal(size=(n, d))
    ytr = rng.integers(0, c, size=n)
    ytr[:c] = np.arange(c)                    # asegura que existan las c clases
    Xte = rng.normal(size=(m, d))
    return Xtr, ytr, Xte


def barrido(titulo, variable, valores, modelo_cls, fase, base):
    """Varía `variable` en `valores` y mide `fase` ('fit' o 'predict')."""
    xs, ts = [], []
    for v in valores:
        p = dict(base); p[variable] = v
        Xtr, ytr, Xte = datos(p["n"], p["d"], p["c"], p["m"])
        modelo = modelo_cls()
        if fase == "fit":
            t = tiempo(lambda: modelo.fit(Xtr, ytr))
        else:
            modelo.fit(Xtr, ytr)
            t = tiempo(lambda: modelo.predict(Xte))
        xs.append(v); ts.append(t)
    pen = pendiente_loglog(xs, ts)
    detalle = "  ".join(f"{v}:{t*1000:.2f}ms" for v, t in zip(xs, ts))
    print(f"{titulo:<44} pendiente={pen:4.2f}   [{detalle}]")


base = dict(n=2000, d=100, c=5, m=500)

print("== 1-NN ==  teoría: fit O(1) (o O(n*d) si se copia) | predict O(m*n*d)")
barrido("predict vs n (muestras de entrenamiento)", "n",
        [500, 1000, 2000, 4000, 8000], UnoNN, "predict", base)
barrido("predict vs d (rasgos)", "d",
        [50, 100, 200, 400, 800], UnoNN, "predict", base)
barrido("predict vs m (muestras de prueba)", "m",
        [250, 500, 1000, 2000, 4000], UnoNN, "predict", base)

print("\n== Distancia Mínima ==  teoría: fit O(n*d) | predict O(m*c*d)")
barrido("fit vs n", "n",
        [2000, 4000, 8000, 16000, 32000], DistanciaMinima, "fit", base)
barrido("predict vs c (clases)", "c",
        [4, 8, 16, 32, 64], DistanciaMinima, "predict",
        dict(n=4000, d=100, c=5, m=4000))
barrido("predict vs n (debe NO depender de n)", "n",
        [500, 2000, 8000, 32000], DistanciaMinima, "predict", base)
