"""
Experimentos Perceptrón Multicapa - MLP (con bibliotecas) - Práctica 3.

Para cada dataset (sintético y WineQT):
  - Validación: Hold-Out 70/30 (estratificado) y K-Fold CV (k=5 y k=10, estratificado).
  - MLPClassifier:
      1) Configuración por default: hidden_layer_sizes=(100,), activation='relu',
         solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200.
      2) Pruebas manuales con parámetros distintos a los default (arquitectura,
         activación, regularización L2 alpha, early_stopping y max_iter).
      3) Mejor configuración encontrada mediante GridSearchCV.
  - La búsqueda de hiperparámetros se realiza SOLO sobre el 70% de entrenamiento;
    el 30% de test se reserva únicamente para la evaluación final.
  - Tiempos de fit/predict y complejidad empírica al variar el número de muestras.

El StandardScaler va dentro de un Pipeline, ajustándose únicamente con los datos de
entrenamiento de cada partición/fold para evitar fuga de información (data leakage).
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
N_FOLDS_MAIN = 5
AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "resultados"

# Rejilla de hiperparámetros para GridSearchCV (ejecutado solo sobre X_tr, y_tr)
PARAM_GRID_MLP = {
    "mlpclassifier__hidden_layer_sizes": [(50,), (100,), (100, 50), (128, 64, 32)],
    "mlpclassifier__activation": ["relu", "tanh"],
    "mlpclassifier__alpha": [0.0001, 0.001, 0.01, 0.05],
    "mlpclassifier__learning_rate_init": [0.001, 0.01],
}


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


def evaluar(nombre_ds, nombre_modelo, params, pipe, X, y, X_tr, X_te, y_tr, y_te, cm_txt, rep_txt):
    """Hold-Out 70/30 + K-Fold CV (k=5 y k=10) sobre todo el dataset."""
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipe.predict(X_te)
    t_pred = time.perf_counter() - t0

    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv5 = cross_validate(pipe, X, y, cv=skf5, scoring=["accuracy", "f1_macro"], n_jobs=-1)

    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    cv10 = cross_validate(pipe, X, y, cv=skf10, scoring=["accuracy", "f1_macro"], n_jobs=-1)

    cm_txt.append(
        f"=== {nombre_ds} | {nombre_modelo} | {params} (Hold-Out 30%) ===\n"
        f"{confusion_matrix(y_te, y_pred)}\n"
    )
    rep_txt.append(
        f"=== {nombre_ds} | {nombre_modelo} | {params} ===\n"
        f"{classification_report(y_te, y_pred, zero_division=0)}\n"
    )

    return {
        "dataset": nombre_ds,
        "modelo": nombre_modelo,
        "parametros": str(params),
        "holdout_acc": accuracy_score(y_te, y_pred),
        "holdout_f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "cv5_acc_media": cv5["test_accuracy"].mean(),
        "cv5_acc_std": cv5["test_accuracy"].std(),
        "cv5_f1_macro_media": cv5["test_f1_macro"].mean(),
        "cv10_acc_media": cv10["test_accuracy"].mean(),
        "cv10_acc_std": cv10["test_accuracy"].std(),
        "cv10_f1_macro_media": cv10["test_f1_macro"].mean(),
        "t_fit_s": t_fit,
        "t_predict_s": t_pred,
    }


def limpiar_params(d):
    return {k.split("__")[-1]: v for k, v in d.items()}


def mejor_tiempo(fn, repeticiones=3):
    mejor = np.inf
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn()
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


def complejidad(nombre_ds, X_tr, y_tr, X_te, modelos):
    """Tiempo de fit/predict al crecer el número de muestras de entrenamiento."""
    filas = []
    for f in (0.25, 0.5, 0.75, 1.0):
        n = max(10, int(len(X_tr) * f))
        for nombre_modelo, pipe in modelos.items():
            p = clone(pipe)
            t_fit = mejor_tiempo(lambda: clone(pipe).fit(X_tr[:n], y_tr[:n]))
            p.fit(X_tr[:n], y_tr[:n])
            t_pred = mejor_tiempo(lambda: p.predict(X_te))
            filas.append({
                "dataset": nombre_ds,
                "modelo": nombre_modelo,
                "fraccion_train": f,
                "n_train": n,
                "d": X_tr.shape[1],
                "t_fit_s": t_fit,
                "t_predict_s": t_pred,
            })
    return filas


def main():
    SALIDA.mkdir(exist_ok=True)
    datasets = cargar_datasets()
    resumen, grid_filas, comp_filas, cm_txt, rep_txt = [], [], [], [], []
    skf_grid = StratifiedKFold(n_splits=N_FOLDS_MAIN, shuffle=True, random_state=SEED)

    for nombre, (X, y) in datasets.items():
        print(f"\n===== {nombre}: {X.shape[0]} muestras, {X.shape[1]} rasgos, clases={np.unique(y)} =====")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        args = (X, y, X_tr, X_te, y_tr, y_te, cm_txt, rep_txt)

        # 1. MLP por defecto
        params_def = {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 200,
        }
        mlp_def = make_pipeline(StandardScaler(), MLPClassifier(random_state=SEED))
        resumen.append(evaluar(nombre, "MLP (default)", params_def, mlp_def, *args))

        # 2. Prueba manual A: Red profunda con tanh y regularización moderada
        params_p1 = {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "tanh",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 1000,
            "early_stopping": True,
        }
        mlp_p1 = make_pipeline(StandardScaler(), MLPClassifier(**params_p1, random_state=SEED))
        resumen.append(evaluar(nombre, "MLP (prueba 1: profunda tanh)", params_p1, mlp_p1, *args))

        # 3. Prueba manual B: Red ancha con ReLU y fuerte regularización L2 (contra ruido)
        params_p2 = {
            "hidden_layer_sizes": (200, 100),
            "activation": "relu",
            "alpha": 0.05,
            "learning_rate_init": 0.005,
            "max_iter": 1000,
            "early_stopping": True,
        }
        mlp_p2 = make_pipeline(StandardScaler(), MLPClassifier(**params_p2, random_state=SEED))
        resumen.append(evaluar(nombre, "MLP (prueba 2: ancha relu L2)", params_p2, mlp_p2, *args))

        # 4. Búsqueda sistemática con GridSearchCV (solo en 70% train)
        print("-> GridSearchCV MLP (buscando mejores hiperparámetros en train)...")
        base_grid = make_pipeline(
            StandardScaler(),
            MLPClassifier(max_iter=1000, early_stopping=True, random_state=SEED),
        )
        gs = GridSearchCV(
            base_grid, PARAM_GRID_MLP, cv=skf_grid, scoring="accuracy", n_jobs=-1
        ).fit(X_tr, y_tr)

        p_best = limpiar_params(gs.best_params_)
        print(f"   Mejores parámetros MLP: {p_best}  (CV en train = {gs.best_score_:.4f})")
        resumen.append(evaluar(nombre, "MLP (mejor GridSearchCV)", p_best, gs.best_estimator_, *args))

        # Guardar ranking completo de GridSearchCV
        cvr = pd.DataFrame(gs.cv_results_)
        for _, r in cvr.iterrows():
            grid_filas.append({
                "dataset": nombre,
                "parametros": str(limpiar_params(r["params"])),
                "hidden_layer_sizes": str(r["params"]["mlpclassifier__hidden_layer_sizes"]),
                "activation": r["params"]["mlpclassifier__activation"],
                "alpha": r["params"]["mlpclassifier__alpha"],
                "learning_rate_init": r["params"]["mlpclassifier__learning_rate_init"],
                "cv_acc_media": r["mean_test_score"],
                "cv_acc_std": r["std_test_score"],
                "t_fit_medio_s": r["mean_fit_time"],
                "ranking": int(r["rank_test_score"]),
            })

        # 5. Análisis de complejidad empírica
        comp_filas += complejidad(
            nombre,
            X_tr,
            y_tr,
            X_te,
            {
                "MLP (default)": make_pipeline(StandardScaler(), MLPClassifier(random_state=SEED)),
                "MLP (mejor GridSearchCV)": gs.best_estimator_,
            },
        )

        print(
            pd.DataFrame([r for r in resumen if r["dataset"] == nombre])
            .drop(columns=["dataset", "parametros", "t_fit_s", "t_predict_s"])
            .round(4)
            .to_string(index=False)
        )

    pd.DataFrame(resumen).to_csv(SALIDA / "resultados_mlp_resumen.csv", index=False)
    pd.DataFrame(grid_filas).sort_values(["dataset", "ranking"]).to_csv(
        SALIDA / "resultados_mlp_gridsearch.csv", index=False
    )
    pd.DataFrame(comp_filas).to_csv(SALIDA / "complejidad_mlp.csv", index=False)
    (SALIDA / "matrices_confusion_mlp.txt").write_text("\n".join(cm_txt), encoding="utf-8")
    (SALIDA / "reportes_clasificacion_mlp.txt").write_text("\n".join(rep_txt), encoding="utf-8")
    print(f"\nListo. Archivos guardados en: {SALIDA}")


if __name__ == "__main__":
    main()