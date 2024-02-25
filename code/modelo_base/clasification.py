"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    def __init__(self, data):
        """"""
        Inicializa la instancia del modelo de clasificación ingenuo.

        Parameters:
            data (pandas.DataFrame): Conjunto de datos que contiene las variables independientes y dependientes.

        Raises:
            ValueError: Se lanza si el conjunto de datos no contiene las columnas esperadas.
        """"""
        self.data = data
        self.features = ['MinTemp', 'MaxTemp', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
        self.target = 'RainTomorrow'

    def clasificacion(self):
        try:
            X = self.data[self.features]
            y = self.data[self.target]

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Calcular la clase mayoritaria
            majority_class = y_train.mode()[0]

            # Predicciones: todas serán la clase mayoritaria
            predictions = np.full_like(y_test, fill_value=majority_class)

            # Calcular métricas de clasificación
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, pos_label=majority_class)
            recall = recall_score(y_test, predictions, pos_label=majority_class)
            f1 = f1_score(y_test, predictions, pos_label=majority_class)

            # Mostrar las métricas utilizando el logger
            print(f"Precisión en el conjunto de prueba: {precision:.2f}")
            print(f"Recall en el conjunto de prueba: {recall:.2f}")
            print(f"F1-score en el conjunto de prueba: {f1:.2f}")
            print(f"Exactitud en el conjunto de prueba: {accuracy:.2f}")



        except Exception as e:
            logger.error(f"Error durante la clasificación: {e}")
            raise ValueError("Error durante la clasificación. Consulta los registros para obtener más detalles.")
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClasificacionModelBase:
    def __init__(self, data_clean):
        """
        Inicializa la instancia del modelo de clasificación.

        Parameters:
            data_clean (pandas.DataFrame): Conjunto de datos limpios que contiene las características relevantes y la etiqueta.
        """
        self.data_clean = data_clean

    def train_and_evaluate(self):
        """
        Entrena y evalúa el modelo base de clasificación.

        Returns:
            dict: Un diccionario que contiene las métricas de rendimiento del modelo.
        """
        try:
            # Paso 1: Preparar los datos
            X = self.data_clean[[column for column in self.data_clean.columns if (column != 'RainTomorrow' or column !='RainfallTomorrow') ]]  # Características relevantes
            y = self.data_clean['RainTomorrow']  # Etiqueta: Lluvia sí/no

            # Paso 2: Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Paso 3: Calcular la clase mayoritaria
            majority_class = y_train.mode()[0]

            # Paso 4: Predicciones: todas serán la clase mayoritaria
            predictions = [majority_class] * len(y_test)

            # Paso 5: Calcular métricas de rendimiento
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, pos_label=majority_class)
            recall = recall_score(y_test, predictions, pos_label=majority_class)
            f1 = f1_score(y_test, predictions, pos_label=majority_class)

            # Paso 6: Devolver métricas de rendimiento como un diccionario
            performance_metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1
            }

            return performance_metrics

        except Exception as e:
            raise ValueError(f"Error durante el entrenamiento y evaluación del modelo: {e}")


