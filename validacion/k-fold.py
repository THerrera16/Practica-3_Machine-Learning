import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

def aplicar_kfold(nombre_archivo, columna_objetivo, k=5):
    # 1. Construir ruta y cargar los dataset´s
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_dataset = os.path.join(ruta_base, 'dataset', nombre_archivo)
    
    df = pd.read_csv(ruta_dataset)
    
    # Separar características (X) y etiquetas (y)
    # Convertimos a arreglos de NumPy (k-fold trabaja con indices numericos) 
    X = df.drop(columns=[columna_objetivo]).values
    y = df[columna_objetivo].values
    
    # n_splits=k define si es de 5 o 10 particiones
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    print(f"Iniciando K-Fold Cross-Validation con K={k}\n")
    
    fold = 1
    # 4. Iterar sobre cada fold (partición)
    for train_index, test_index in skf.split(X, y):
        # Separar los datos para este fold específico
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print(f"--- Fold {fold} ---")
        print(f"Entrenamiento: {len(X_train)} muestras | Prueba: {len(X_test)} muestras")
        fold += 1
    print("-" * 30)

if __name__ == "__main__":
    
    # Evaluar Dataset Sintético con K=5
    print("=== EVALUACIÓN DATASET SINTÉTICO ===")
    aplicar_kfold('mdlon_sintetico.csv', 'T', k=5)
    
    # Evaluar Dataset Benchmark con K=10 (puedes cambiarlo a 5 si prefieres)
    print("\n=== EVALUACIÓN DATASET BENCHMARK ===")
    aplicar_kfold('WineQT_benchmark.csv', 'quality', k=5)