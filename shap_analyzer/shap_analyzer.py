import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class SHAPAnalyzer:
    def __init__(self, model, data):
        """
        Inicializa el analizador SHAP.

        Args:
            model: Un modelo ya entrenado que se pueda usar con SHAP.
            data: Los datos sobre los cuales se calcularán los valores SHAP.
        """
        self.model = model
        self.data = data

        # Crea un objeto Explainer para obtener los valores SHAP
        self.explainer = shap.Explainer(model)
        self.shap_values = self.explainer.shap_values(data)

    def summary_plot(self):
        """
        Muestra un gráfico de resumen SHAP.
        """
        shap.summary_plot(self.shap_values, self.data)

    def force_plot(self, index=0):
        """
        Muestra un gráfico de fuerza SHAP para una observación específica.

        Args:
            index (int): El índice de la observación en los datos.
        """
        shap.force_plot(self.explainer.expected_value, self.shap_values[index, :], self.data.iloc[index, :])

    def waterfall_plot(self, index=0):
        """
        Muestra un gráfico de cascada SHAP para una observación específica.

        Args:
            index (int): El índice de la observación en los datos.
        """
        shap.waterfall_plot(self.explainer.expected_value, self.shap_values[index, :], self.data.iloc[index, :])

    def dependence_plot(self, feature_name):
        """
        Muestra un gráfico de dependencia SHAP para una característica específica.

        Args:
            feature_name (str): El nombre de la característica.
        """
        shap.dependence_plot(feature_name, self.shap_values, self.data)
