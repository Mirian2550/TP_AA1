import streamlit as st
import joblib
import pandas as pd
import os
path_dir = os.path.dirname(os.path.abspath(__file__))
# Cargar los modelos y los datos limpios
print(path_dir)
pipeline_regresion = joblib.load('weather_regression.joblib')
pipeline_clasificacion = joblib.load('weather_classification.joblib')
data = pd.read_csv('data/weatherAUS_clean.csv')

# Obtener las columnas esperadas por el modelo
columnas_esperadas = data.columns.drop(['RainTomorrow', 'RainfallTomorrow'])

# Sliders para las características
sliders = {}
for col in columnas_esperadas:
    if col == 'RainToday':
        sliders[col] = st.slider(col, int(data[col].min()), int(data[col].max()))
    else:
        sliders[col] = st.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))

# Crear un DataFrame con los valores para predecir
data_para_predecir = pd.DataFrame([sliders])

# Realizar la predicción de clasificación
prediccion_clasificacion = pipeline_clasificacion.predict(data_para_predecir)
# Realizar la predicción de regresión si clasificación indica que lloverá
prediccion_regresion = None
if prediccion_clasificacion[0] == 1:
    prediccion_regresion = pipeline_regresion.predict(data_para_predecir)

# Mostrar la predicción de clasificación
st.write('Predicción Clasificación:', 'Lloverá' if prediccion_clasificacion[0] == 1 else 'No lloverá')

# Mostrar la predicción de regresión si clasificación indica que lloverá
if prediccion_clasificacion[0] == 1:
    st.write(f'Predicción Regresión (cantidad de lluvia): {prediccion_regresion[0]} mm' if prediccion_regresion is not None else 'No disponible')
