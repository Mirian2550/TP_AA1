import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.exceptions import ConvergenceWarning

# Configurar el logger para manejar las advertencias de convergencia
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Desactivar las advertencias de convergencia para mantener limpia la salida
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ClasificationModelBase:
    """
    Clase base para modelos de clasificación.

    Parameters:
        data (pandas.DataFrame): Conjunto de datos que contiene las variables independientes y dependientes.
    """

    def __init__(self, data):
        """
        Inicializa la instancia del modelo de clasificación.

        Parameters:
            data (pandas.DataFrame): Conjunto de datos que contiene las variables independientes y dependientes.

        Raises:
            ValueError: Se lanza si el conjunto de datos no contiene las columnas esperadas.
        """
        expected_columns = ['MinTemp', 'MaxTemp', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
                            'RainToday', 'RainTomorrow', 'RainfallTomorrow', 'WindGustDir_numerico']

        if not set(expected_columns).issubset(data.columns):
            raise ValueError(f"El conjunto de datos debe contener las columnas: {expected_columns}")

        self.data = data
        self.features = expected_columns

    def clasificacion(self):
        try:
            X = self.data[self.features]
            y = self.data['RainTomorrow']

            # Escalar características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Aumentar el número máximo de iteraciones
            self.model = LogisticRegression(max_iter=1000, solver='lbfgs')

            # Entrenar el modelo de clasificación
            self.model.fit(X_train, y_train)

            # Realizar predicciones en el conjunto de prueba
            predictions = self.model.predict(X_test)

            # Calcular métricas de clasificación
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            # Mostrar las métricas utilizando el logger
            print(f"Precisión en el conjunto de prueba: {precision:.2f}")
            print(f"Recall en el conjunto de prueba: {recall:.2f}")
            print(f"F1-score en el conjunto de prueba: {f1:.2f}")
            print(f"Exactitud en el conjunto de prueba: {accuracy:.2f}")

            # Devolver resultados en el formato esperado
            return X_test, y_test, predictions, self.model

        except Exception as e:
            logger.error(f"Error durante la clasificación: {e}")
            raise ValueError("Error durante la clasificación. Consulta los registros para obtener más detalles.") from e