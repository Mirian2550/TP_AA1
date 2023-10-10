import pandas as pd
data = pd.read_csv('weatherAUS.csv')

#print(data['RainfallTomorrow'].describe())
data.fillna(method='ffill', inplace=True)

print("Valores nulos en RainfallTomorrow:", data['RainfallTomorrow'].isnull().sum())
"""
Date Location RainTomorrow RainfallTomorrow
"""
ciudades = [ 'Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport' ]
#data = data[data['Location'].isin(ciudades)]
data.drop(data[~data['Location'].isin(ciudades)].index, inplace=True)


