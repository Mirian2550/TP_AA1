import numpy as np
import pandas as pd
from matplotlib.widgets import Lasso
from scipy.stats import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/weatherAUS.csv')

print(data.head)
#[145412 rows x 25 columns]>

cities_of_interest = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
filtered_data = data[data['Location'].isin(cities_of_interest)]
#print(filtered_data)
#[15986 rows x 25 columns]


"""
‘RainTomorrow’ ‘RainfallTomorrow’ # son las principales
Date, Location #no tocar

datos a tener en cuenta:
145412 rows x 25 columns 
145458 cantidad de datos segun indice del csv #eliminamos esta columna?

cantidad de datos nulos por columnas:
MinTemp              1484 #reemplazar por dia anterior
MaxTemp              1253 #reemplazar por dia anterior
Rainfall             3260 #reemplazar por 0
Evaporation         62754 #eliminar
Sunshine            69796 #eliminar
WindGustDir         10316 #eliminar
WindGustSpeed       10253 #eliminar
WindDir9am          10562 #eliminar
WindDir3pm           4226 #no sirve?
WindSpeed9am         1767 #?
WindSpeed3pm         3061 #?
Humidity3pm          4505 #reemplazar por dia anterior
Pressure9am         15061 #eliminar
Pressure3pm         15024 #eliminar
Cloud9am            55870 #eliminar
Cloud3pm            59336 #eliminar
Temp9am              1766 #reemplazar por dia anterior
Temp3pm              3607 #reemplazar por dia anterior
RainToday            3260 #llovera hoy reemplazar por?
RainTomorrow         3259 #llovera mañana reemplazar por?
RainfallTomorrow     3259 #lluvia mañana? ver que variable/s afecta/n a esta condicion y las 2 de arriba
# eliminar por % de cantidad de valores nulos

WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, RainToday #eliminar por ser str?

"""
#print(data['RainfallTomorrow'].describe())
#data.fillna(method='ffill', inplace=True)

#print("Valores nulos en RainfallTomorrow:", data['RainfallTomorrow'].isnull().sum())
"""
Date Location RainTomorrow RainfallTomorrow
"""
#ciudades = [ 'Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport' ]
#data = data[data['Location'].isin(ciudades)]
#data.drop(data[~data['Location'].isin(ciudades)].index, inplace=True)


# Ver la cantidad de datos faltantes por columna
missing_data = data.isnull().sum()

# Mostrar las columnas con datos faltantes y la cantidad correspondiente
#print(missing_data[missing_data > 0])

#Reemplazar valores nulos teniendo en cuenta los datos
"""
data['MinTemp'].fillna(data['MinTemp'].mean(), inplace=True)
data['MaxTemp'].fillna(data['MaxTemp'].median(), inplace=True)
data['Rainfall'].fillna(data['Rainfall'].mean(), inplace=True)
data['Evaporation'].fillna(data['Evaporation'].mean(), inplace=True)
data['Sunshine'].fillna(data['Sunshine'].mean(), inplace=True)
data['WindGustSpeed'].fillna(data['WindGustSpeed'].mean(), inplace=True)
data['WindSpeed9am'].fillna(data['WindSpeed9am'].mean(), inplace=True)
data['WindSpeed3pm'].fillna(data['WindSpeed3pm'].mean(), inplace=True)
data['Humidity9am'].fillna(data['Humidity9am'].median(), inplace=True)
data['Humidity3pm'].fillna(data['Humidity3pm'].median(), inplace=True)
data['Pressure9am'].fillna(data['Pressure9am'].mean(), inplace=True)
data['Pressure3pm'].fillna(data['Pressure3pm'].mean(), inplace=True)
data['Cloud9am'].fillna(data['Cloud9am'].mean(), inplace=True)
data['Cloud3pm'].fillna(data['Cloud3pm'].mean(), inplace=True)
data['Temp9am'].fillna(data['Temp9am'].mean(), inplace=True)
data['Temp3pm'].fillna(data['Temp3pm'].mean(), inplace=True)
data['WindGustDir'].fillna(data['WindGustDir'].mode()[0], inplace=True)
data['WindDir9am'].fillna(data['WindDir9am'].mode()[0], inplace=True)
data['WindDir3pm'].fillna(data['WindDir3pm'].mode()[0], inplace=True)
data['RainToday'].fillna(data['RainToday'].mode()[0], inplace=True)
data['RainTomorrow'].fillna(data['RainTomorrow'].mode()[0], inplace=True)

print(data.head())
"""