"""
Inciso 2 - Experimentos de Distancia Mínima y 1-NN (implementados desde cero).

Usa EXACTAMENTE las mismas particiones que los scripts del equipo:
  * Hold Out 70/30 : train_test_split(test_size=0.3, random_state=42, stratify=y)
  * K-Fold         : StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
(scikit-learn se usa únicamente para generar las particiones, igual que
holdout.py y k-fold.py; los clasificadores y las métricas son propios.)

Genera:
  resultados/resultados_inciso2.csv           tabla con todos los resultados
  resultados/matrices_confusion_holdout.txt   matrices de confusión (Hold Out)
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from clasificadores_desde_cero import (
    DistanciaMinima, UnoNN, estandarizar,
    exactitud, f1_macro, matriz_confusion,
)

RUTA_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_RESULTADOS = os.path.join(RUTA_BASE, "inciso2", "resultados")
K_FOLDS = 5          # el equipo usa k=5 en k-fold.py
SEMILLA = 42


# ----------------------------------------------------------------------
# Carga de datasets
# ----------------------------------------------------------------------
def cargar(nombre):
    """Devuelve X (ndarray), y (ndarray) del dataset pedido."""
    if nombre == "Sintético (mdlon)":
        df = pd.read_csv(os.path.join(RUTA_BASE, "dataset", "mdlon_sintetico.csv"))
        return df.drop(columns=["T"]).values.astype(float), df["T"].values

    df = pd.read_csv(os.path.join(RUTA_BASE, "dataset", "WineQT_benchmark.csv"))
    # "Id" es solo un identificador de fila, NO es un rasgo del vino.
    X = df.drop(columns=["quality", "Id"]).values.astype(float)
    q = df["quality"].values
    if nombre == "Wine (6 clases)":
        return X, q
    if nombre == "Wine (3 clases)":
        # baja (<=5) -> 0, media (6) -> 1, alta (>=7) -> 2
        return X, np.where(q <= 5, 0, np.where(q == 6, 1, 2))
    raise ValueError(nombre)


MODELOS = {"Distancia Mínima": DistanciaMinima, "1-NN": UnoNN}
DATASETS = ["Sintético (mdlon)", "Wine (6 clases)", "Wine (3 clases)"]


# ----------------------------------------------------------------------
# Un entrenamiento + una predicción, con tiempos
# ----------------------------------------------------------------------
def evaluar(clase_modelo, Xtr, ytr, Xte, yte, clases, escalar):
    if escalar:
        Xtr, Xte = estandarizar(Xtr, Xte)
    modelo = clase_modelo()

    t0 = time.perf_counter()
    modelo.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = modelo.predict(Xte)
    t_pred = time.perf_counter() - t0

    return {
        "exactitud": exactitud(yte, pred),
        "f1_macro": f1_macro(yte, pred, clases),
        "t_fit": t_fit,
        "t_pred": t_pred,
        "confusion": matriz_confusion(yte, pred, clases),
    }


# ----------------------------------------------------------------------
# Ciclo principal
# ----------------------------------------------------------------------
def main():
    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    filas = []
    texto_confusion = []

    for ds in DATASETS:
        X, y = cargar(ds)
        clases = np.unique(y)
        print(f"\n=== {ds}: {X.shape[0]} muestras, {X.shape[1]} rasgos, "
              f"{len(clases)} clases ===")

        # Hold Out 70/30
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=SEMILLA, stratify=y
        )
        # K-Fold estratificado
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMILLA)
        folds = list(skf.split(X, y))

        for nombre_modelo, clase in MODELOS.items():
            for escalar in (False, True):
                etiqueta_esc = "z-score" if escalar else "sin escalar"

                # ---- Hold Out
                r = evaluar(clase, Xtr, ytr, Xte, yte, clases, escalar)
                filas.append({
                    "dataset": ds, "modelo": nombre_modelo,
                    "preprocesamiento": etiqueta_esc, "validacion": "Hold Out 70/30",
                    "exactitud": r["exactitud"], "desv_exactitud": np.nan,
                    "f1_macro": r["f1_macro"],
                    "t_fit_s": r["t_fit"], "t_pred_s": r["t_pred"],
                })
                texto_confusion.append(
                    f"{ds} | {nombre_modelo} | {etiqueta_esc} | clases={list(clases)}\n"
                    f"{r['confusion']}\n"
                )

                # ---- K-Fold
                res = [evaluar(clase, X[a], y[a], X[b], y[b], clases, escalar)
                       for a, b in folds]
                filas.append({
                    "dataset": ds, "modelo": nombre_modelo,
                    "preprocesamiento": etiqueta_esc,
                    "validacion": f"K-Fold (k={K_FOLDS})",
                    "exactitud": np.mean([x["exactitud"] for x in res]),
                    "desv_exactitud": np.std([x["exactitud"] for x in res]),
                    "f1_macro": np.mean([x["f1_macro"] for x in res]),
                    "t_fit_s": np.mean([x["t_fit"] for x in res]),
                    "t_pred_s": np.mean([x["t_pred"] for x in res]),
                })

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(RUTA_RESULTADOS, "resultados_inciso2.csv"),
              index=False, float_format="%.6f")
    with open(os.path.join(RUTA_RESULTADOS, "matrices_confusion_holdout.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(texto_confusion))

    # Vista rápida en consola
    vista = df.copy()
    vista["exactitud"] = vista["exactitud"].round(4)
    vista["desv_exactitud"] = vista["desv_exactitud"].round(4)
    vista["f1_macro"] = vista["f1_macro"].round(4)
    vista["t_fit_s"] = vista["t_fit_s"].map(lambda v: f"{v:.5f}")
    vista["t_pred_s"] = vista["t_pred_s"].map(lambda v: f"{v:.5f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n", vista.to_string(index=False))


if __name__ == "__main__":
    main()
