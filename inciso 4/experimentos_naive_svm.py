"""
Experimentos Naïve Bayes y SVM (con bibliotecas) - Práctica 3.

Para cada dataset (sintético y WineQT):
  - Validación: Hold-Out 70/30 (estratificado) y K-Fold CV (k=5, estratificado).
  - Naïve Bayes (GaussianNB): default y con var_smoothing ajustado.
  - SVM: default (SVC(): kernel rbf, C=1, gamma='scale') vs. la mejor configuración
    encontrada con GridSearchCV (kernels linear / rbf / poly / sigmoid, C, gamma, degree).
  - La búsqueda de parámetros se hace SOLO con el 70% de entrenamiento; el 30% de test
    se usa una única vez para evaluar.
  - Tiempos de fit/predict y complejidad empírica al variar el número de muestras.

El StandardScaler va dentro de un Pipeline, por lo que se ajusta únicamente con los datos de
entrenamiento de cada partición/fold (sin fuga de información).

"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
N_FOLDS = 5
AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "resultados"


INCLUIR_POLY = True

PARAM_GRID_SVM = [
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.001, 0.01, 0.1]},
    {"svc__kernel": ["sigmoid"], "svc__C": [0.1, 1, 10], "svc__gamma": ["scale"]},
]
if INCLUIR_POLY:
    PARAM_GRID_SVM.append({"svc__kernel": ["poly"], "svc__degree": [2, 3],
                           "svc__C": [0.1, 1, 10], "svc__gamma": ["scale"]})

PARAM_GRID_NB = {"gaussiannb__var_smoothing": [1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1]}


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
    """Hold-Out (entrena en 70%, evalúa en 30%) + K-Fold CV sobre todo el dataset."""
    t0 = time.perf_counter(); pipe.fit(X_tr, y_tr); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); y_pred = pipe.predict(X_te); t_pred = time.perf_counter() - t0

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv = cross_validate(pipe, X, y, cv=skf, scoring=["accuracy", "f1_macro"])  

    cm_txt.append(f"=== {nombre_ds} | {nombre_modelo} | {params} (Hold-Out 30%) ===\n"
                  f"{confusion_matrix(y_te, y_pred)}\n")
    rep_txt.append(f"=== {nombre_ds} | {nombre_modelo} | {params} ===\n"
                   f"{classification_report(y_te, y_pred, zero_division=0)}\n")
    return {
        "dataset": nombre_ds, "modelo": nombre_modelo, "parametros": str(params),
        "holdout_acc": accuracy_score(y_te, y_pred),
        "holdout_f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
        f"cv{N_FOLDS}_acc_media": cv["test_accuracy"].mean(),
        f"cv{N_FOLDS}_acc_std": cv["test_accuracy"].std(),
        f"cv{N_FOLDS}_f1_macro_media": cv["test_f1_macro"].mean(),
        "t_fit_s": t_fit, "t_predict_s": t_pred,
    }


def limpiar_params(d):
    return {k.split("__")[-1]: v for k, v in d.items()}


def mejor_tiempo(fn, repeticiones=3):
    mejor = np.inf
    for _ in range(repeticiones):
        t0 = time.perf_counter(); fn(); mejor = min(mejor, time.perf_counter() - t0)
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
            filas.append({"dataset": nombre_ds, "modelo": nombre_modelo, "fraccion_train": f,
                          "n_train": n, "d": X_tr.shape[1], "t_fit_s": t_fit, "t_predict_s": t_pred})
    return filas



def main():
    SALIDA.mkdir(exist_ok=True)
    datasets = cargar_datasets()
    resumen, grid_filas, comp_filas, cm_txt, rep_txt = [], [], [], [], []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for nombre, (X, y) in datasets.items():
        print(f"\n===== {nombre}: {X.shape[0]} muestras, {X.shape[1]} rasgos, clases={np.unique(y)} =====")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        args = (X, y, X_tr, X_te, y_tr, y_te, cm_txt, rep_txt)

        
        nb_def = make_pipeline(StandardScaler(), GaussianNB())
        resumen.append(evaluar(nombre, "Naive Bayes (default)", {}, nb_def, *args))

        print("-> GridSearch Naive Bayes (var_smoothing)...")
        gs_nb = GridSearchCV(make_pipeline(StandardScaler(), GaussianNB()), PARAM_GRID_NB,
                             cv=skf, scoring="accuracy", n_jobs=-1).fit(X_tr, y_tr)
        p_nb = limpiar_params(gs_nb.best_params_)
        resumen.append(evaluar(nombre, "Naive Bayes (var_smoothing ajustado)", p_nb, gs_nb.best_estimator_, *args))

        
        svm_def = make_pipeline(StandardScaler(), SVC(random_state=SEED))
        resumen.append(evaluar(nombre, "SVM (default)", {"kernel": "rbf", "C": 1.0, "gamma": "scale"}, svm_def, *args))

        
        print("-> GridSearchCV SVM (puede tardar un poco)...")
        gs = GridSearchCV(make_pipeline(StandardScaler(), SVC(random_state=SEED)), PARAM_GRID_SVM,
                          cv=skf, scoring="accuracy", n_jobs=-1).fit(X_tr, y_tr)
        p_svm = limpiar_params(gs.best_params_)
        print(f"   Mejores parámetros SVM: {p_svm}  (CV en train = {gs.best_score_:.4f})")
        resumen.append(evaluar(nombre, "SVM (mejor GridSearchCV)", p_svm, gs.best_estimator_, *args))

        
        cvr = pd.DataFrame(gs.cv_results_)
        for _, r in cvr.iterrows():
            grid_filas.append({"dataset": nombre, "parametros": str(limpiar_params(r["params"])),
                               "kernel": r["params"]["svc__kernel"],
                               "C": r["params"].get("svc__C"), "gamma": r["params"].get("svc__gamma"),
                               "degree": r["params"].get("svc__degree"),
                               "cv_acc_media": r["mean_test_score"], "cv_acc_std": r["std_test_score"],
                               "t_fit_medio_s": r["mean_fit_time"], "ranking": int(r["rank_test_score"])})

        
        comp_filas += complejidad(nombre, X_tr, y_tr, X_te,
                                  {"Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
                                   "SVM (default)": make_pipeline(StandardScaler(), SVC(random_state=SEED)),
                                   "SVM (mejor)": gs.best_estimator_})

        print(pd.DataFrame([r for r in resumen if r["dataset"] == nombre])
              .drop(columns=["dataset", "t_fit_s", "t_predict_s"]).round(4).to_string(index=False))

    pd.DataFrame(resumen).to_csv(SALIDA / "resultados_naive_svm_resumen.csv", index=False)
    pd.DataFrame(grid_filas).sort_values(["dataset", "ranking"]).to_csv(SALIDA / "resultados_svm_gridsearch.csv", index=False)
    pd.DataFrame(comp_filas).to_csv(SALIDA / "complejidad_naive_svm.csv", index=False)
    (SALIDA / "matrices_confusion_naive_svm.txt").write_text("\n".join(cm_txt), encoding="utf-8")
    (SALIDA / "reportes_clasificacion.txt").write_text("\n".join(rep_txt), encoding="utf-8")
    print(f"\nListo. Archivos guardados en: {SALIDA}")


if __name__ == "__main__":     
    main()
