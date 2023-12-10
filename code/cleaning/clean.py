import pandas as pd
import logging


class Clean:

    def __init__(self, data_source):
        self.data = pd.read_csv(data_source)
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


    def _fill_missing_values(self):
        ciudades = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
        self.data_clean = self.data[self.data['Location'].isin(ciudades)]
        self.data_clean = self.data_clean.drop('WindGustSpeed', axis=1)

        # Llena valores faltantes de manera más general
        self.data_clean = self.data_clean.groupby('Location').apply(lambda group: group.ffill().bfill())

        self.data_clean['Rainfall'] = self.data_clean['Rainfall'].fillna(0)
        self.data_clean['RainToday'] = self.data_clean['RainToday'].fillna('No')

        median_rainfall = self.data_clean['RainfallTomorrow'].median()
        self.data_clean['RainfallTomorrow'] = self.data_clean['RainfallTomorrow'].fillna(median_rainfall)

    def _encode_categorical_columns(self):
        self.data_clean = pd.get_dummies(self.data_clean, columns=['RainToday'], drop_first=True)
        self.data_clean = pd.get_dummies(self.data_clean, columns=['RainTomorrow'], drop_first=True)

    def _process_numerical_columns(self):
        columns_to_round = [
            'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
            'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
            'Temp9am', 'Temp3pm'
        ]
        columns_negative_to_0 = ['Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm', 'Cloud9am', 'Cloud3pm']
        columns_to_cap = ['Humidity9am', 'Humidity3pm']

        # Realiza operaciones en columnas numéricas
        for column in columns_to_round:
            if column in self.data_clean.columns:
                self.data_clean[column] = self.data_clean[column].round(1)

        for column in columns_negative_to_0:
            if column in self.data_clean.columns:
                self.data_clean[column] = self.data_clean[column].clip(lower=0)

        for column in columns_to_cap:
            if column in self.data_clean.columns:
                self.data_clean[column] = self.data_clean[column].clip(upper=100)

        columns_tmp = [
            'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
            'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
            'Temp9am', 'Temp3pm'
        ]

        for column in columns_tmp:
            q1 = self.data_clean[column].quantile(0.25)
            q3 = self.data_clean[column].quantile(0.75)
            iqr = q3 - q1
            lower_limit = round(q1 - 1.5 * iqr, 1)
            upper_limit = round(q3 + 1.5 * iqr, 1)
            self.data_clean = self.data_clean[
                (self.data_clean[column] >= lower_limit) & (self.data_clean[column] <= upper_limit)
                ]

    def process(self):
        try:
            self._fill_missing_values()
            self._encode_categorical_columns()
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

            ciudades = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
            datos_filtrados = self.data[self.data['Location'].isin(ciudades)]
            datos_filtrados = datos_filtrados.drop('WindGustSpeed', axis=1)
            datos_filtrados = datos_filtrados.drop(['Unnamed: 0'], axis=1) # Dado que la primer columna es el índice la eliminamos

            datos_filtrados.loc[:, 'MinTemp'] = datos_filtrados.groupby('Location')['MinTemp'].ffill()
            datos_filtrados.loc[:, 'MaxTemp'] = datos_filtrados.groupby('Location')['MaxTemp'].ffill()
            datos_filtrados.loc[:, 'Rainfall'] = datos_filtrados['Rainfall'].fillna(0)
            datos_filtrados.loc[:, 'Temp9am'] = datos_filtrados['Temp9am'].ffill()
            datos_filtrados.loc[:, 'Temp3pm'] = datos_filtrados['Temp3pm'].ffill()
            datos_filtrados.loc[:, 'Humidity3pm'] = datos_filtrados['Humidity3pm'].ffill()
            datos_filtrados.loc[:, 'Cloud3pm'] = datos_filtrados['Cloud3pm'].ffill()
            datos_filtrados.loc[:, 'Evaporation'] = datos_filtrados['Evaporation'].ffill()
            datos_filtrados.loc[:, 'Sunshine'] = datos_filtrados.groupby('Location')['Sunshine'].ffill()
            datos_filtrados['WindGustDir'] = datos_filtrados['WindGustDir'].fillna(datos_filtrados['WindDir9am'].combine_first(datos_filtrados['WindDir3pm']))
            datos_filtrados = datos_filtrados.drop(['WindDir9am', 'WindDir3pm'], axis=1)
            direccion_mapping = {
                'N': 1, 'NNE': 1, 'NE': 1, 'ENE': 1,
                'E': 2, 'ESE': 2, 'SE': 2, 'SSE': 2,
                'S': 3, 'SSW': 3, 'SW': 3, 'WSW': 3,
                'W': 4, 'WNW': 4, 'NW': 4, 'NNW': 4
            }

            datos_filtrados['WindGustDir'] = datos_filtrados['WindGustDir'].map(direccion_mapping)

            datos_filtrados.loc[:, 'WindSpeed9am'] = datos_filtrados['WindSpeed9am'].ffill()
            datos_filtrados.loc[:, 'WindSpeed3pm'] = datos_filtrados['WindSpeed3pm'].ffill()
            datos_filtrados.loc[:, 'Humidity9am'] = datos_filtrados['Humidity9am'].ffill()
            datos_filtrados.loc[:, 'Pressure9am'] = datos_filtrados['Pressure9am'].ffill()
            datos_filtrados.loc[:, 'Pressure3pm'] = datos_filtrados['Pressure3pm'].ffill()
            datos_filtrados.loc[:, 'Cloud9am'] = datos_filtrados['Cloud9am'].ffill()
            datos_filtrados.loc[:, 'RainToday'] = datos_filtrados['RainToday'].fillna('No')
            datos_filtrados.loc[:, 'RainTomorrow'] = datos_filtrados['RainTomorrow'].ffill()

            median_rainfall = datos_filtrados['RainfallTomorrow'].median()
            datos_filtrados.loc[:, 'RainfallTomorrow'] = datos_filtrados['RainfallTomorrow'].fillna(median_rainfall)
            self.data_clean = datos_filtrados
            columnas_nulas = self.data_clean.columns[self.data_clean.isnull().any()]

            data_rain_tomorrow = pd.get_dummies(self.data_clean["RainTomorrow"], drop_first=True)
            data_rain_tomorrow = data_rain_tomorrow.astype(int)
            self.data_clean["RainTomorrow"] = data_rain_tomorrow
            data_rain_today = pd.get_dummies(self.data_clean["RainToday"], drop_first=True)

            data_rain_today = data_rain_today.astype(int)
            self.data_clean["RainToday"] = data_rain_today

        except Exception as e:
            self.logger.error(f"Error en la limpieza de datos: {str(e)}")
            raise ValueError(f"Error en la limpieza de datos: {str(e)}")

    def _process(self):
        """
        Realiza el preprocesamiento de los datos limpios, incluyendo la detección y manejo de valores atípicos.

        redondear a 1 decimal
        Evaporation, sunshine, WindSpeed9am, WindSpeed3pm, Cloud9am, Cloud3pm negativos poner en 0
        Humidity que no supere 100.0
        """
        try:
            self.data_clean = self.data_clean.drop('Unnamed: 0', axis=1)
            columns_to_round = [
                'MinTemp',
                'MaxTemp',
                'Evaporation',
                'Sunshine',
                'WindSpeed9am',
                'WindSpeed3pm',
                'Humidity9am',
                'Humidity3pm',
                'Pressure9am',
                'Pressure3pm',
                'Cloud9am',
                'Cloud3pm',
                'Temp9am',
                'Temp3pm'
            ]
            columns_negative_to_0 = ['Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm', 'Cloud9am', 'Cloud3pm']
            columns_to_cap = ['Humidity9am', 'Humidity3pm']
            for column in columns_to_round:
                if column in self.data_clean.columns:
                    self.data_clean[column] = self.data_clean[column].round(1)

            for column in columns_negative_to_0:
                if column in self.data_clean.columns:
                    self.data_clean[column] = self.data_clean[column].clip(lower=0)

            for column in columns_to_cap:
                if column in self.data_clean.columns:
                    self.data_clean[column] = self.data_clean[column].clip(upper=100)

            columns_tmp = ['MinTemp', 'MaxTemp',
                           'Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                           'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                           'Temp9am', 'Temp3pm']

            for column in columns_tmp:
                q1 = self.data_clean[column].quantile(0.25)
                q3 = self.data_clean[column].quantile(0.75)
                iqr = q3 - q1
                lower_limit = round(q1 - 1.5 * iqr, 1)
                upper_limit = round(q3 + 1.5 * iqr, 1)
                self.data_clean = self.data_clean[
                    (self.data_clean[column] >= lower_limit) & (self.data_clean[column] <= upper_limit)
                    ]
        except Exception as e:
            self.logger.error(f"Error en el preprocesamiento de datos: {str(e)}")
            raise ValueError(f"Error en el preprocesamiento de datos: {str(e)}")

