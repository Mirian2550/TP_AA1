import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class RegresionModelBase:
    def __init__(self, data):
        """
        Inicializa la instancia del modelo de regresión lineal base.

        Parameters:
            data (pandas.DataFrame): Conjunto de datos que contiene las variables independientes y dependientes.

        Raises:
            ValueError: Se lanza si el conjunto de datos no contiene las columnas esperadas.
        """
        expected_columns = ['RainfallTomorrow']

        if not set(expected_columns).issubset(data.columns):
            raise ValueError(f"El conjunto de datos debe contener las columnas: {expected_columns}")

        self.data = data

    def train(self):
        """
        Entrena el modelo de regresión lineal simple.

        Returns:
            LinearRegression: Modelo entrenado.
            float: Error cuadrático medio (MSE) en el conjunto de prueba.
            float: Coeficiente de determinación (R^2) en el conjunto de prueba.
        """
        try:
            # Paso 1: Preparar los datos
            X = self.data[['RainToday']]  # Variable independiente: Temperatura a las 3pm
            y = self.data['RainfallTomorrow']  # Variable dependiente: Cantidad de lluvia mañana

            # Paso 2: Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Paso 3: Crear y entrenar el modelo de regresión lineal simple
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Paso 4: Evaluar el modelo
            predictions = model.predict(X_test)

            # Calcular métricas de rendimiento
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            return model, mse, r2

        except Exception as e:
            raise ValueError(f"Error durante el entrenamiento del modelo: {e}")
