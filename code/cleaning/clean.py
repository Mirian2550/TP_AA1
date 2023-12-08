import pandas as pd
import logging


class Clean:

    def __init__(self, data_source):
        self.data = data_source
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

            columnas_nulas = self.data_clean.columns[self.data_clean.isnull().any()]

            if columnas_nulas.empty:
                print("No hay columnas con valores nulos en data_clean.")
            else:
                print("Columnas con valores nulos en data_clean:")
        except Exception as e:
            self.logger.error(f"Error en el preprocesamiento de datos: {str(e)}")
            raise ValueError(f"Error en el preprocesamiento de datos: {str(e)}")
