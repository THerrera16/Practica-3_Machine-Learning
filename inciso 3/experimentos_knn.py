"""
Experimentos K-NN (desde cero) - Práctica 3.

Qué hace, para cada dataset (sintético y WineQT):
  1. Hold-Out 70/30 (estratificado) y K-Fold CV (k=5, estratificado).
  2. Prueba K-NN con parámetros distintos al default:
        k ∈ {1,3,5,7,9,11,15,21,31}, métrica ∈ {euclidean, manhattan}, pesos ∈ {uniform, distance}
  3. Elige la mejor configuración con CV *dentro del 70% de entrenamiento*
     (el 30% de test nunca se usa para elegir parámetros) y compara contra el default.
  4. Verifica la implementación contra sklearn (solo como comprobación).
  5. Mide tiempos de fit/predict al variar n (muestras) y d (rasgos) -> complejidad empírica.

El escalado (StandardScaler) se ajusta SOLO con los datos de entrenamiento de cada partición,
para evitar fuga de información (data leakage).

"""
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier  # SOLO para verificar la implementación
from sklearn.preprocessing import StandardScaler

from knn_desde_cero import KNNDesdeCero

SEED = 42
N_FOLDS = 5                      # el enunciado permite k=5 o k=10
KS = [1, 3, 5, 7, 9, 11, 15, 21, 31]
METRICAS = ["euclidean", "manhattan"]
PESOS = ["uniform", "distance"]
CONFIGS = [(k, m, w) for m, w, k in product(METRICAS, PESOS, KS)]
DEFAULT = (5, "euclidean", "uniform")   # default de sklearn
UNO_NN = (1, "euclidean", "uniform")

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "resultados"


def buscar_carpeta_dataset():
    for base in (AQUI, AQUI.parent, Path.cwd(), Path.cwd().parent):
        if (base / "dataset").is_dir():
            return base / "dataset"
    raise FileNotFoundError("No encontré la carpeta 'dataset/' (busqué junto al script y un nivel arriba).")


def cargar_datasets():
    ruta = buscar_carpeta_dataset()
    df_s = pd.read_csv(ruta / "mdlon_sintetico.csv")
    X_s, y_s = df_s.drop(columns=["T"]).values, df_s["T"].values

    df_w = pd.read_csv(ruta / "WineQT_benchmark.csv")
    y_w = df_w["quality"].apply(lambda q: 0 if q <= 5 else (1 if q == 6 else 2)).values
    X_w = df_w.drop(columns=[c for c in ("quality", "Id") if c in df_w.columns]).values
    return {"Sintetico (mdlon)": (X_s, y_s), "Benchmark (WineQT)": (X_w, y_w)}


def escalar(X_tr, X_te):
    sc = StandardScaler().fit(X_tr)           
    return sc.transform(X_tr), sc.transform(X_te)



def predicciones_todas(X_tr, y_tr, X_te):
    """Escala, y para cada métrica calcula los vecinos UNA vez (hasta max(KS)) y
    reutiliza esos vecinos para todos los k y tipos de peso."""
    X_tr, X_te = escalar(X_tr, X_te)
    preds = {}
    for metrica in METRICAS:
        modelo = KNNDesdeCero(metric=metrica).fit(X_tr, y_tr)
        dist, idx = modelo.kneighbors(X_te, max(KS))
        for peso, k in product(PESOS, KS):
            preds[(k, metrica, peso)] = modelo.votar(dist, idx, k, peso)
    return preds


def cv_todas(X, y):
    # K-Fold estratificado: devuelve {config: (lista_acc, lista_f1)}.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    res = {c: ([], []) for c in CONFIGS}
    for tr, te in skf.split(X, y):
        preds = predicciones_todas(X[tr], y[tr], X[te])
        for c, p in preds.items():
            res[c][0].append(accuracy_score(y[te], p))
            res[c][1].append(f1_score(y[te], p, average="macro", zero_division=0))
    return res


def mejor_tiempo(fn, repeticiones=3):
    mejor = np.inf
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn()
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


def medir_complejidad(nombre, X_tr, y_tr, X_te):
    X_tr, X_te = escalar(X_tr, X_te)
    filas = []
    n, d = X_tr.shape
    k, metrica, peso = DEFAULT
    for variable, fracs in (("n_train", [0.25, 0.5, 0.75, 1.0]), ("n_rasgos", [0.25, 0.5, 0.75, 1.0])):
        for f in fracs:
            n_f = max(k, int(n * f)) if variable == "n_train" else n
            d_f = max(1, int(d * f)) if variable == "n_rasgos" else d
            Xa, ya, Xq = X_tr[:n_f, :d_f], y_tr[:n_f], X_te[:, :d_f]
            modelo = KNNDesdeCero(k, metrica, peso)
            t_fit = mejor_tiempo(lambda: modelo.fit(Xa, ya))
            modelo.fit(Xa, ya)
            t_pred = mejor_tiempo(lambda: modelo.predict(Xq))
            filas.append({"dataset": nombre, "variable": variable, "fraccion": f,
                          "n_train": n_f, "n_test": len(Xq), "d": d_f,
                          "t_fit_s": t_fit, "t_predict_total_s": t_pred,
                          "t_predict_por_muestra_ms": 1000 * t_pred / len(Xq)})
    return filas

def main():
    SALIDA.mkdir(exist_ok=True)
    datasets = cargar_datasets()
    filas_todas, filas_resumen, filas_comp, texto_cm = [], [], [], []

    for nombre, (X, y) in datasets.items():
        print(f"\n===== {nombre}: {X.shape[0]} muestras, {X.shape[1]} rasgos, clases={np.unique(y)} =====")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y)

        # 1) Hold-Out
        preds_ho = predicciones_todas(X_tr, y_tr, X_te)
        # 2) K-Fold sobre TODO el dataset (validación pedida en el enunciado)
        cv_all = cv_todas(X, y)
        # 3) K-Fold solo sobre entrenamiento (para ELEGIR parámetros sin tocar el test)
        cv_tr = cv_todas(X_tr, y_tr)

        mejor = max(CONFIGS, key=lambda c: np.mean(cv_tr[c][0]))  # empate -> k menor (orden de CONFIGS)

        for c in CONFIGS:
            k, m, w = c
            filas_todas.append({
                "dataset": nombre, "k": k, "metrica": m, "pesos": w,
                "es_default": c == DEFAULT, "seleccionada_por_cv": c == mejor,
                "holdout_acc": accuracy_score(y_te, preds_ho[c]),
                "holdout_f1_macro": f1_score(y_te, preds_ho[c], average="macro", zero_division=0),
                f"cv{N_FOLDS}_acc_media": np.mean(cv_all[c][0]),
                f"cv{N_FOLDS}_acc_std": np.std(cv_all[c][0]),
                f"cv{N_FOLDS}_f1_macro_media": np.mean(cv_all[c][1]),
                "cv_train_acc_media(seleccion)": np.mean(cv_tr[c][0]),
            })

        # Resumen: 1-NN, K-NN default y K-NN mejor
        for etiqueta, c in (("1-NN", UNO_NN), ("K-NN default (k=5)", DEFAULT), ("K-NN mejor (CV en train)", mejor)):
            filas_resumen.append({
                "dataset": nombre, "modelo": etiqueta, "k": c[0], "metrica": c[1], "pesos": c[2],
                "holdout_acc": accuracy_score(y_te, preds_ho[c]),
                "holdout_f1_macro": f1_score(y_te, preds_ho[c], average="macro", zero_division=0),
                f"cv{N_FOLDS}_acc_media": np.mean(cv_all[c][0]),
                f"cv{N_FOLDS}_acc_std": np.std(cv_all[c][0]),
            })
            texto_cm.append(f"=== {nombre} | {etiqueta} | k={c[0]}, {c[1]}, {c[2]} (Hold-Out 30%) ===\n"
                            f"{confusion_matrix(y_te, preds_ho[c])}\n")

        # Verificación contra sklearn (mismos datos escalados)
        Xs_tr, Xs_te = escalar(X_tr, X_te)
        k, m, w = mejor
        sk = KNeighborsClassifier(n_neighbors=k, metric=m, weights=w).fit(Xs_tr, y_tr).predict(Xs_te)
        coincide = np.mean(sk == preds_ho[mejor])
        print(f"Verificación vs sklearn (config {mejor}): coinciden {100 * coincide:.2f}% de las predicciones "
              f"(diferencias mínimas solo por empates de distancia)")

        # Complejidad empírica
        filas_comp += medir_complejidad(nombre, X_tr, y_tr, X_te)

        # Impresión en consola
        res = pd.DataFrame([f for f in filas_resumen if f["dataset"] == nombre]).drop(columns="dataset")
        print(res.round(4).to_string(index=False))

    pd.DataFrame(filas_todas).to_csv(SALIDA / "resultados_knn_todas_las_configs.csv", index=False)
    pd.DataFrame(filas_resumen).to_csv(SALIDA / "resultados_knn_resumen.csv", index=False)
    pd.DataFrame(filas_comp).to_csv(SALIDA / "complejidad_knn.csv", index=False)
    (SALIDA / "matrices_confusion_knn.txt").write_text("\n".join(texto_cm), encoding="utf-8")
    print(f"\nListo. Archivos guardados en: {SALIDA}")


if __name__ == "__main__":
    main()
