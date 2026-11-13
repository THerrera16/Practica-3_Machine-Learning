"""
Verificación (NO es parte de la implementación pedida):
compara las predicciones de nuestras clases desde cero contra las de
scikit-learn sobre la misma partición Hold Out 70/30.
  DistanciaMinima  <->  NearestCentroid
  UnoNN            <->  KNeighborsClassifier(n_neighbors=1)
También contrasta las métricas propias con las de sklearn.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from clasificadores_desde_cero import (
    DistanciaMinima, UnoNN, estandarizar, exactitud, f1_macro,
)
from experimentos_inciso2 import DATASETS, SEMILLA, cargar

print(f"{'dataset':<20}{'modelo':<18}{'escala':<13}{'coinciden':>10}"
      f"{'dif. exactitud':>16}{'dif. F1':>10}")
for ds in DATASETS:
    X, y = cargar(ds)
    clases = np.unique(y)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=SEMILLA, stratify=y)
    for escalar in (False, True):
        a, b = (estandarizar(Xtr, Xte) if escalar else (Xtr, Xte))
        pares = [
            ("Distancia Mínima", DistanciaMinima(), NearestCentroid()),
            ("1-NN", UnoNN(), KNeighborsClassifier(n_neighbors=1)),
        ]
        for nombre, propio, ref in pares:
            p_propio = propio.fit(a, ytr).predict(b)
            p_ref = ref.fit(a, ytr).predict(b)
            coinciden = np.mean(p_propio == p_ref)
            d_acc = abs(exactitud(yte, p_propio) - accuracy_score(yte, p_propio))
            d_f1 = abs(f1_macro(yte, p_propio, clases)
                       - f1_score(yte, p_propio, average="macro", labels=clases,
                                  zero_division=0))
            print(f"{ds:<20}{nombre:<18}{'z-score' if escalar else 'sin escalar':<13}"
                  f"{coinciden:>9.2%}{d_acc:>16.1e}{d_f1:>10.1e}")
