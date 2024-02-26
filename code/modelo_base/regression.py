from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class RegresionModelBase:
    def __init__(self):
        self.model = 'Base Regresion'

    def train(self, _x_train, _x_test, _y_train_classification, _y_test_classification):
        """
        Entrena el modelo de regresión lineal simple.

        Parameters:
            _x_train (pandas.DataFrame): Características de entrenamiento.
            _x_test (pandas.DataFrame): Características de prueba.
            _y_train_classification (pandas.Series): Etiquetas de entrenamiento.
            _y_test_classification (pandas.Series): Etiquetas de prueba.

        Returns:
            LinearRegression: Modelo entrenado.
            float: Error cuadrático medio (MSE) en el conjunto de prueba.
            float: Coeficiente de determinación (R^2) en el conjunto de prueba.
        """
        try:
            # Paso 1: Crear y entrenar el modelo de regresión lineal simple
            model = LinearRegression()
            model.fit(_x_train, _y_train_classification)

            # Paso 2: Evaluar el modelo
            predictions = model.predict(_x_test)

            # Calcular métricas de rendimiento
            mse = mean_squared_error(_y_test_classification, predictions)
            r2 = r2_score(_y_test_classification, predictions)

            return model, mse, r2

        except Exception as e:
            raise ValueError(f"Error durante el entrenamiento del modelo: {e}")
