import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class RegresionModelBase:
    """
    Clase base para modelos de regresión.

    Parameters:
        data (pandas.DataFrame): Conjunto de datos que contiene las variables independientes y dependientes.
    """

    def __init__(self, data):
        """
        Inicializa la instancia del modelo de regresión.

        Parameters:
            data (pandas.DataFrame): Conjunto de datos que contiene las variables independientes y dependientes.

        Raises:
            ValueError: Se lanza si el conjunto de datos no contiene las columnas esperadas.
        """
        expected_columns = ['MinTemp', 'MaxTemp', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
                            'RainToday', 'RainfallTomorrow']

        if not set(expected_columns).issubset(data.columns):
            raise ValueError(f"El conjunto de datos debe contener las columnas: {expected_columns}")

        self.data = data
        self.model = LinearRegression()
        self.features = expected_columns

    def regresion(self):
        """
        Entrena el modelo de regresión, realiza predicciones en el conjunto de prueba y muestra métricas de rendimiento.

        Returns:
            tuple: Una tupla que contiene x_test, y_test, y_pred y el modelo entrenado.

        Raises:
            ValueError: Se lanza si hay problemas durante la regresión.
        """
        try:
            X = self.data[self.features]
            y = self.data['Rainfall']

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Definir los hiperparámetros a ajustar con Grid Search
            param_grid = {'fit_intercept': [True, False], 'positive': [True, False]}


            # Crear el objeto GridSearchCV
            grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring='neg_mean_squared_error',
                                       cv=5)

            # Entrenar el modelo con búsqueda de hiperparámetros
            grid_search.fit(X_train, y_train)

            # Obtener el mejor modelo después de la búsqueda de hiperparámetros
            best_model = grid_search.best_estimator_

            # Realizar predicciones en el conjunto de prueba con el mejor modelo
            predictions = best_model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            # Mostrar las métricas utilizando el logger o la función print
            print(f"Error Cuadrático Medio en el conjunto de prueba: {mse}")
            print(f"Error Absoluto Medio en el conjunto de prueba: {mae}")
            print(f"Raíz del Error Cuadrático Medio en el conjunto de prueba: {rmse}")
            print(f"Coeficiente de Determinación (R^2) en el conjunto de prueba: {r2}")
            # Devolver resultados en el formato esperado
            return X_test, y_test, predictions, best_model

        except Exception as e:
            logger.error(f"Error durante la regresión: {e}")
            raise ValueError("Error durante la regresión. Consulta los registros para obtener más detalles.") from e
