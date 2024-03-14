import pandas as pd
import numpy as np

# Función para calcular los coeficientes de regresión manualmente
def regresion_manual(X, y):
    # Agregar una columna de unos para el término independiente
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Calcular los coeficientes utilizando la fórmula de la pseudo inversa
    coeficientes = np.linalg.pinv(X.T @ X) @ X.T @ y

    return coeficientes  

# Función para predecir los valores de y
def predecir(X, coeficientes):
    Xm = np.hstack([np.ones((X.shape[0], 1)), X])
    
    return Xm @ coeficientes

# Calcular métricas de evaluación manualmente
def rmse(y_true, y_pred):
    error = y_true - y_pred
    return np.sqrt(np.mean(error ** 2))

def r2F(y_true, y_pred):
    numerador = ((y_true - y_pred) ** 2).sum()
    denominador = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (numerador / denominador)

# Función para ajustar el modelo y evaluarlo
def ajustar_evaluar_modelo(X, y):
    coeficientes = regresion_manual(X, y)
    y_pred = predecir(X, coeficientes)
    r2_ = r2F(y, y_pred)
    rmse_val = rmse(y, y_pred)
    return coeficientes, y_pred, r2_, rmse_val

opcion = int(input())
# Cargar los datos
data = pd.read_csv('Mediciones.csv')

# Definir las columnas de características (X) y la columna de objetivo (y)
if opcion == 1:
    print("Número de filas y columnas:", data.shape)
    
    # Seleccionar las características (variables independientes) y el objetivo
    caracteristicas = ['VTI_F', 'BPM', 'PEEP', 'VTE_F']
    objetivo = ['Pasos']
    
    X = data[caracteristicas]
    y = data[objetivo]
    
    print("Características:")
    print(X.head())
    print("Objetivo:")
    print(y.head())

elif opcion == 2:
    # Modelo completo solo con VTI_F
    X = data[['VTI_F']]
    y = data['Pasos']
    coef = regresion_manual(X, y)
    print("Coeficientes del modelo VTI_F:", coef)

elif opcion == 3:
    # Modelo completo solo con VTI_F, calcular métricas
    X = data[['VTI_F']]
    y = data['Pasos']
    coef = regresion_manual(X, y)
    y_pred = predecir(X, coef)
    print("Ejemplo de predicción para los primeros 3 valores:")
    print(y.head(3).to_numpy().flatten(), y_pred[:3])
    r2_ = r2F(y, y_pred)
    rmse_val = rmse(y, y_pred)
    print("R2:", r2_)
    print("RMSE:", rmse_val)

elif opcion == 4:
    # Modelo completo solo con VTI_F, ajustar y evaluar
    X_todo = data[['VTI_F']]
    y_todo = data['Pasos']
    coeficientes_todo, y_pred_todo, r2_todo, rmse_todo = ajustar_evaluar_modelo(X_todo, y_todo)
    print("Modelo completo con VTI_F - R2:", r2_todo)
    print("Modelo completo con VTI_F - RMSE:", rmse_todo)

elif opcion == 5:
    # Combinaciones de características de los modelos solicitados
    models = {
        'Modelo_1': ['VTI_F'],
        'Modelo_2': ['VTI_F', 'BPM'],
        'Modelo_3': ['VTI_F', 'PEEP'],
        'Modelo_4': ['VTI_F', 'PEEP', 'BPM'],
        'Modelo_5': ['VTI_F', 'PEEP', 'BPM', 'VTE_F']
    }
    for nombre_modelo, lista_caracteristicas in models.items():
        X = data[lista_caracteristicas]
        y = data['Pasos']
        coeficientes, y_pred, r2, rmse_val = ajustar_evaluar_modelo(X, y)
        print(nombre_modelo, " - R2:", r2, " - RMSE:", rmse_val)

elif opcion == 6:
    # Modelos para cada combinación de PEEP y BPM
    valores_peep_unicos = data['PEEP'].unique()
    valores_bpm_unicos = data['BPM'].unique()
    
    print("Valores únicos de PEEP:", valores_peep_unicos)
    print("Valores únicos de BPM:", valores_bpm_unicos)
    
    predicciones_totales = []
    for peep in valores_peep_unicos:
        for bpm in valores_bpm_unicos:
            datos_subset = data[(data['PEEP'] == peep) & (data['BPM'] == bpm)]
            X_subset = datos_subset[['VTI_F']]
            y_subset = datos_subset['Pasos']
            coeficientes_subset, y_pred_subset, r2_subset, rmse_subset = ajustar_evaluar_modelo(X_subset, y_subset)
            print("PEEP:", peep, "BPM:", bpm, " - R2:", r2_subset, " - RMSE:", rmse_subset)
            predicciones_totales.append(y_pred_subset)
    
    predicciones_concatenadas = np.concatenate(predicciones_totales)
    y = data['Pasos']
    r2_global = r2F(y, predicciones_concatenadas)
    rmse_global = rmse(y, predicciones_concatenadas)
    print('Global - R2:', r2_global, ' - RMSE:', rmse_global)