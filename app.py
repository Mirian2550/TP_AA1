import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Cargar el modelo y los datos limpios
pipeline_entrenado = joblib.load('weather.joblib')
data = pd.read_csv('data/weatherAUS_clean.csv')

# Obtener las columnas esperadas por el modelo
columnas_esperadas = data.columns  # Excluir la columna objetivo ('RainTomorrow')

# Sliders para las características
sliders = {}
for col in columnas_esperadas:
    sliders[col] = st.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))

# Crear un DataFrame con los valores para predecir
data_para_predecir = pd.DataFrame([sliders])

# Realizar la predicción
prediccion = pipeline_entrenado.predict(data_para_predecir)

# Mostrar la predicción
st.write('Predicción:', 'Lloverá' if prediccion[0] == 1 else 'No lloverá')
