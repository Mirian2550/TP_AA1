import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import RobustScaler

class Clean:

    def __init__(self, data):
        self.data = data
        self.data_clean = None
        self.logger = self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger("clean")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


    def _process_numerical_columns(self):
        self.data_clean = self.data
        columns_negative_to_0 = ['Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm', 'Cloud9am', 'Cloud3pm']
        columns_to_cap = ['Humidity9am', 'Humidity3pm']

        for column in columns_negative_to_0:
            if column in self.data_clean.columns:
                self.data_clean[column] = self.data_clean[column].clip(lower=0)

        for column in columns_to_cap:
            if column in self.data_clean.columns:
                self.data_clean[column] = self.data_clean[column].clip(upper=100)
        
        # RobustScaler
        for column in self.data_clean.select_dtypes(include=np.number).columns:
            scaler = RobustScaler()

            self.data_clean[column] = scaler.fit_transform(self.data_clean[column].values.reshape(-1, 1)).flatten()


    def process(self):
        try:
            # Eliminación de registros con datos nulos/faltantes en variables booleanas
            # Se eliminan porque constituyen menos de un 6% de los datos
            # y no afecta la representatividad de los datos
            self.data.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
            self._process_numerical_columns()
            self._clean()
            return self.data_clean
        except Exception as e:
            self.logger.error(f"Error en la procesamiento de datos: {str(e)}")
            raise RuntimeError(f"Error en la procesamiento de datos: {str(e)}")

    def _clean(self):
        """
        Limpia los datos cargados desde el archivo CSV y prepara los datos limpios para su uso.
        """
        try:
            datos_filtrados = self.data_clean
            datos_filtrados = datos_filtrados.drop('WindGustSpeed', axis=1) # Notamos que la correlación para nuestras variables de interés es baja por lo tanto decidimos eliminarla.
            datos_filtrados = datos_filtrados.drop(['Unnamed: 0'], axis=1)  # Dado que la primer columna es el índice la eliminamos
            datos_filtrados.loc[:, 'Rainfall'] = datos_filtrados['Rainfall'].fillna(0) # Reemplazamos por 0 los valores nulos
            # Reemplazamos los valores nulos de las siguientes variables con el valor anterior de la misma ciudad:
            datos_filtrados.loc[:, 'MinTemp'] = datos_filtrados.groupby('Location')['MinTemp'].ffill()
            datos_filtrados.loc[:, 'MaxTemp'] = datos_filtrados.groupby('Location')['MaxTemp'].ffill()
            datos_filtrados.loc[:, 'Temp9am'] = datos_filtrados.groupby('Location')['Temp9am'].ffill()
            datos_filtrados.loc[:, 'Temp3pm'] = datos_filtrados.groupby('Location')['Temp3pm'].ffill()
            datos_filtrados.loc[:, 'Humidity3pm'] = datos_filtrados.groupby('Location')['Humidity3pm'].ffill()
            datos_filtrados.loc[:, 'Cloud3pm'] = datos_filtrados.groupby('Location')['Cloud3pm'].ffill()
            datos_filtrados.loc[:, 'Evaporation'] = datos_filtrados.groupby('Location')['Evaporation'].ffill()
            datos_filtrados.loc[:, 'Sunshine'] = datos_filtrados.groupby('Location')['Sunshine'].ffill()
            datos_filtrados.loc[:, 'WindGustDir'] = datos_filtrados.groupby('Location')['WindGustDir'].ffill()
            datos_filtrados.loc[:, 'WindDir9am'] = datos_filtrados.groupby('Location')['WindDir9am'].ffill()
            datos_filtrados.loc[:, 'WindDir3pm'] = datos_filtrados.groupby('Location')['WindDir3pm'].ffill()
            datos_filtrados.loc[:, 'WindSpeed9am'] = datos_filtrados.groupby('Location')['WindSpeed9am'].ffill()
            datos_filtrados.loc[:, 'WindSpeed3pm'] = datos_filtrados.groupby('Location')['WindSpeed3pm'].ffill()
            datos_filtrados.loc[:, 'Humidity9am'] = datos_filtrados.groupby('Location')['Humidity9am'].ffill()
            datos_filtrados.loc[:, 'Pressure9am'] = datos_filtrados.groupby('Location')['Pressure9am'].ffill()
            datos_filtrados.loc[:, 'Pressure3pm'] = datos_filtrados.groupby('Location')['Pressure3pm'].ffill()
            datos_filtrados.loc[:, 'Cloud9am'] = datos_filtrados.groupby('Location')['Cloud9am'].ffill()
            # Realizamos un mapeo numérico de la dirección del viento para poder realizar el gráfico de correlación:
            direccion_mapping = {
                'E': 0, 'ENE': 22.5, 'NE': 45, 'NNE': 67.5,
                'N': 90, 'NNW': 112.5, 'NW': 135, 'WNW': 157.5,
                'W': 180, 'WSW': 202.5, 'SW': 225, 'SSW': 247.5,
                'S': 270, 'SSE': 292.5, 'SE': 315, 'ESE': 337.5,
                }
            datos_filtrados['WindGustDir'] = datos_filtrados['WindGustDir'].map(direccion_mapping)
            datos_filtrados['WindDir9am'] = datos_filtrados['WindDir9am'].map(direccion_mapping)
            datos_filtrados['WindDir3pm'] = datos_filtrados['WindDir3pm'].map(direccion_mapping)
            datos_filtrados.loc[:, 'RainToday'] = datos_filtrados['RainToday'].fillna(0)
            datos_filtrados.loc[:, 'RainTomorrow'] = datos_filtrados['RainTomorrow'].fillna(0) 
            median_rainfall = datos_filtrados['RainfallTomorrow'].median() 
            datos_filtrados.loc[:, 'RainfallTomorrow'] = datos_filtrados['RainfallTomorrow'].fillna(median_rainfall) # Reemplazamos por la mediana para que no modifique el promedio

            self.data_clean = datos_filtrados

            columnas_nulas = self.data_clean.columns[self.data_clean.isnull().any()]

            self.data_clean.to_csv('data/weatherAUS_clean.csv', index=False)
            print("Archivo guardado exitosamente en 'data/weatherAUS_clean.csv'")

        except Exception as e:
            self.logger.error(f"Error en la limpieza de datos: {str(e)}")
            raise ValueError(f"Error en la limpieza de datos: {str(e)}")
