import pandas as pd
from sklearn.model_selection import train_test_split
import os

def cargar_y_dividir_dataset(nombre_archivo, columna_objetivo):
    # Construir la ruta para llegar a los dataset´s
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_dataset = os.path.join(ruta_base, 'dataset', nombre_archivo)
    
    # Cargar el dataset´s 
    df = pd.read_csv(ruta_dataset)
    
    # Separar características (X) y etiquetas (y)
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo]
    
    # Aplicar Hold Out 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    
    archivo_sintetico = 'mdlon_sintetico.csv' 
    columna_clase_sintetico = 'T'
    
    print("--- Resultados Dataset Sintético ---")
    X_train_sint, X_test_sint, y_train_sint, y_test_sint = cargar_y_dividir_dataset(archivo_sintetico, columna_clase_sintetico)
    print(f"Muestras de entrenamiento (70%): {len(X_train_sint)}")
    print(f"Muestras de prueba (30%): {len(X_test_sint)}\n")
    

    archivo_benchmark = 'WineQT_benchmark.csv' 
    columna_clase_benchmark = 'quality'
    
    print("--- Resultados Dataset Benchmark ---")
    X_train_bench, X_test_bench, y_train_bench, y_test_bench = cargar_y_dividir_dataset(archivo_benchmark, columna_clase_benchmark)
    print(f"Muestras de entrenamiento (70%): {len(X_train_bench)}")
    print(f"Muestras de prueba (30%): {len(X_test_bench)}")
    print(f"Muestras de prueba (30%): {len(X_test_bench)}")