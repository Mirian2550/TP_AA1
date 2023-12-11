import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.title('Predicción de lluvia')

pipeline_entrenado = joblib.load('weather.joblib')
data = pd.read_csv('data/weatherAUS_clean.csv')

# Sliders para las características
MinTemp = st.slider('MinTemp', float(data.MinTemp.min()), float(data.MinTemp.max()), float(data.MinTemp.mean()))
MaxTemp = st.slider('MaxTemp', float(data.MaxTemp.min()), float(data.MaxTemp.max()), float(data.MaxTemp.mean()))
Rainfall = st.slider('Rainfall', float(data.Rainfall.min()), float(data.Rainfall.max()), float(data.Rainfall.mean()))
Evaporation = st.slider('Evaporation', float(data.Evaporation.min()), float(data.Evaporation.max()), float(data.Evaporation.mean()))
Sunshine = st.slider('Sunshine', float(data.Sunshine.min()), float(data.Sunshine.max()), float(data.Sunshine.mean()))
WindGustDir = st.slider('WindGustDir', float(data.WindGustDir.min()), float(data.WindGustDir.max()), float(data.WindGustDir.mean()))
WindSpeed9am = st.slider('WindSpeed9am', float(data.WindSpeed9am.min()), float(data.WindSpeed9am.max()), float(data.WindSpeed9am.mean()))
WindSpeed3pm = st.slider('WindSpeed3pm', float(data.WindSpeed3pm.min()), float(data.WindSpeed3pm.max()), float(data.WindSpeed3pm.mean()))
Humidity9am = st.slider('Humidity9am', float(data.Humidity9am.min()), float(data.Humidity9am.max()), float(data.Humidity9am.mean()))
Humidity3pm = st.slider('Humidity3pm', float(data.Humidity3pm.min()), float(data.Humidity3pm.max()), float(data.Humidity3pm.mean()))
Pressure9am = st.slider('Pressure9am', float(data.Pressure9am.min()), float(data.Pressure9am.max()), float(data.Pressure9am.mean()))
Pressure3pm = st.slider('Pressure3pm', float(data.Pressure3pm.min()), float(data.Pressure3pm.max()), float(data.Pressure3pm.mean()))
Cloud9am = st.slider('Cloud9am', float(data.Cloud9am.min()), float(data.Cloud9am.max()), float(data.Cloud9am.mean()))
Cloud3pm = st.slider('Cloud3pm', float(data.Cloud3pm.min()), float(data.Cloud3pm.max()), float(data.Cloud3pm.mean()))
Temp9am = st.slider('Temp9am', float(data.Temp9am.min()), float(data.Temp9am.max()), float(data.Temp9am.mean()))
Temp3pm = st.slider('Temp3pm', float(data.Temp3pm.min()), float(data.Temp3pm.max()), float(data.Temp3pm.mean()))
RainToday = st.slider('RainToday', float(data.RainToday.min()), float(data.RainToday.max()), float(data.RainToday.mean()))
RainTomorrow = st.slider('RainTomorrow', float(data.RainTomorrow.min()), float(data.RainTomorrow.max()), float(data.RainTomorrow.mean()))

# Crear un DataFrame con los valores para predecir
data_para_predecir = pd.DataFrame([[MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday]]) 

# Realizar la predicción
prediccion = pipeline_entrenado.predict(data_para_predecir)

# Mostrar la predicción
st.write('Predicción:', prediccion)
