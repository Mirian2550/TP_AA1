import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

def evaluar_modelo(y_true, y_pred):
    """
    Evalúa un modelo de regresión y calcula diversas métricas.

    Args:
        y_true (array-like): Valores reales u observados.
        y_pred (array-like): Valores predichos por el modelo.

    Returns:
        dict: Un diccionario que contiene las métricas calculadas, incluyendo MSE, R^2, RMSE, MAE y MAPE.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "MSE": mse,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }

class ModeloPrediccionLluvia:
    """
    Inicializa una instancia de la clase ModeloPrediccionLluvia.
    Args:
        file (str): Ruta del archivo CSV que contiene los datos.
    """
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.data_clean = None
        self.logger = logging.getLogger("ModeloPrediccionLluvia")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def limpiar_datos(self):
        """
        Limpia los datos cargados desde el archivo CSV y prepara los datos limpios para su uso.
        """
        try:
            ciudades = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
            datos_filtrados = self.data[self.data['Location'].isin(ciudades)]
            datos_filtrados = datos_filtrados.drop('WindGustSpeed', axis=1)

            datos_filtrados.loc[:, 'MinTemp'] = datos_filtrados.groupby('Location')['MinTemp'].ffill()
            datos_filtrados.loc[:, 'MaxTemp'] = datos_filtrados.groupby('Location')['MaxTemp'].ffill()
            datos_filtrados.loc[:, 'Rainfall'] = datos_filtrados['Rainfall'].fillna(0)
            datos_filtrados.loc[:, 'Temp9am'] = datos_filtrados['Temp9am'].ffill()
            datos_filtrados.loc[:, 'Temp3pm'] = datos_filtrados['Temp3pm'].ffill()
            datos_filtrados.loc[:, 'Humidity3pm'] = datos_filtrados['Humidity3pm'].ffill()
            datos_filtrados.loc[:, 'Cloud3pm'] = datos_filtrados['Cloud3pm'].ffill()
            datos_filtrados.loc[:, 'Evaporation'] = datos_filtrados['Evaporation'].ffill()
            datos_filtrados.loc[:, 'Sunshine'] = datos_filtrados.groupby('Location')['Sunshine'].ffill()
            datos_filtrados.loc[:, 'WindGustDir'] = datos_filtrados['WindGustDir'].fillna('sin datos')
            datos_filtrados.loc[:, 'WindDir9am'] = datos_filtrados['WindDir9am'].fillna('sin datos')
            datos_filtrados.loc[:, 'WindDir3pm'] = datos_filtrados['WindDir3pm'].fillna('sin datos')
            datos_filtrados.loc[:, 'WindSpeed9am'] = datos_filtrados['WindSpeed9am'].ffill()
            datos_filtrados.loc[:, 'WindSpeed3pm'] = datos_filtrados['WindSpeed3pm'].ffill()
            datos_filtrados.loc[:, 'Humidity9am'] = datos_filtrados['Humidity9am'].ffill()
            datos_filtrados.loc[:, 'Pressure9am'] = datos_filtrados['Pressure9am'].ffill()
            datos_filtrados.loc[:, 'Pressure3pm'] = datos_filtrados['Pressure3pm'].ffill()
            datos_filtrados.loc[:, 'Cloud9am'] = datos_filtrados['Cloud9am'].ffill()
            datos_filtrados.loc[:, 'RainToday'] = datos_filtrados['RainToday'].fillna('sin datos')
            datos_filtrados.loc[:, 'RainTomorrow'] = datos_filtrados['RainTomorrow'].ffill()

            median_rainfall = datos_filtrados['RainfallTomorrow'].median()
            datos_filtrados.loc[:, 'RainfallTomorrow'] = datos_filtrados['RainfallTomorrow'].fillna(median_rainfall)
            self.data_clean = datos_filtrados
            columnas_nulas = self.data_clean.columns[self.data_clean.isnull().any()]
            if columnas_nulas.empty:
                print("No hay columnas con valores nulos en data_clean.")
            else:
                print("Columnas con valores nulos en data_clean:")
                print(columnas_nulas)
        except Exception as e:
            self.logger.error(f"Error en la limpieza de datos: {str(e)}")
            raise ValueError(f"Error en la limpieza de datos: {str(e)}")

    def visualizar_datos(self):
        """
        Visualiza los datos limpios y muestra gráficos e información estadística.
        """
        try:
            columns_tmp = ['MinTemp', 'MaxTemp', 'Rainfall',
                           'Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                           'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                           'Temp9am', 'Temp3pm', 'RainfallTomorrow']

            rows = 4
            cols = 4
            fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
            fig.subplots_adjust(hspace=0.8, wspace=0.5)  # Ajustar espacio vertical y horizontal
            # Iterar sobre las columnas y graficar en subplots
            for i, column in enumerate(columns_tmp):
                sns.histplot(self.data_clean[column], bins=30, edgecolor='black', ax=axes[i // cols, i % cols])
                axes[i // cols, i % cols].set_title(f'Histograma de {column}')
                axes[i // cols, i % cols].set_xlabel(column)
                axes[i // cols, i % cols].set_ylabel('Frecuencia')

            # Ajustar el diseño general
            plt.tight_layout()
            plt.show()

            # Boxplots
            fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
            fig.subplots_adjust(hspace=0.8, wspace=0.5)

            for i, column in enumerate(columns_tmp):
                sns.boxplot(x=self.data_clean[column], ax=axes[i // cols, i % cols])
                axes[i // cols, i % cols].set_title(f'Diagrama de Caja de {column}')
                axes[i // cols, i % cols].set_xlabel(column)
                axes[i // cols, i % cols].set_ylabel('Valor')

            # Ajustar el diseño general
            plt.tight_layout()
            plt.show()

            # Scatterplots
            fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
            fig.subplots_adjust(hspace=0.8, wspace=0.5)

            for i, column in enumerate(columns_tmp):
                sns.scatterplot(x=column, y='RainfallTomorrow', data=self.data_clean, ax=axes[i // cols, i % cols])
                axes[i // cols, i % cols].set_title(f'Scatterplot entre {column} y RainfallTomorrow')
                axes[i // cols, i % cols].set_xlabel(column)
                axes[i // cols, i % cols].set_ylabel('RainfallTomorrow')

            # Ajustar el diseño general
            plt.tight_layout()
            plt.show()

            matriz_correlacion = self.data_clean[columns_tmp].corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Matriz de Correlación (sin Date)')
            plt.show()
            conteo_clases = self.data_clean['RainTomorrow'].value_counts()

            # Calcula el balance
            porcentaje_clase_positiva = (conteo_clases['Yes'] / len(self.data_clean)) * 100
            porcentaje_clase_negativa = (conteo_clases['No'] / len(self.data_clean)) * 100

            print(f'Porcentaje de clase "Yes" (Lloverá): {porcentaje_clase_positiva:.2f}%')
            print(f'Porcentaje de clase "No" (No lloverá): {porcentaje_clase_negativa:.2f}%')
        except Exception as e:
            self.logger.error(f"Error en la visualización de datos: {str(e)}")
            raise ValueError(f"Error en la visualización de datos: {str(e)}")

    def preprocesar_datos(self):
        """
        Realiza el preprocesamiento de los datos limpios, incluyendo la detección y manejo de valores atípicos.
        """
        try:
            columns_tmp = ['MinTemp', 'MaxTemp',
                           'Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                           'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                           'Temp9am', 'Temp3pm']

            for column in columns_tmp:
                # Calcular el rango intercuartílico (IQR)
                Q1 = self.data_clean[column].quantile(0.25)
                Q3 = self.data_clean[column].quantile(0.75)
                IQR = Q3 - Q1

                # Definir los límites para identificar valores atípicos
                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR

                # Reemplazar valores atípicos
                self.data_clean.loc[
                    (self.data_clean[column] < lower_limit) | (self.data_clean[column] > upper_limit), column] = \
                    self.data_clean[column].mean()
        except Exception as e:
            self.logger.error(f"Error en el preprocesamiento de datos: {str(e)}")
            raise ValueError(f"Error en el preprocesamiento de datos: {str(e)}")

    def entrenar_regresion_lineal(self, normalize=True):
        """
        Entrena un modelo de regresión lineal utilizando los datos preprocesados.
        Args:
            normalize (bool, optional): Indica si se deben normalizar las características. Por defecto, True.

        Returns:
            tuple: Un par de arrays, y_true y y_pred, que contienen los valores reales y predichos, respectivamente.
        """
        try:
            columnas_caracteristicas = ['Rainfall', 'Humidity3pm', 'Cloud3pm']
            variable_objetivo = 'RainfallTomorrow'
            X = self.data_clean[columnas_caracteristicas]
            y = self.data_clean[variable_objetivo]

            modelo = LinearRegression()
            if normalize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Entrena el modelo en todos los datos
            modelo.fit(X, y)
            if normalize:
                # Si las características se normalizaron, podemos inspeccionar los coeficientes
                coeficientes = modelo.coef_
                print("Coeficientes de características normalizadas:")
                for caracteristica, coef in zip(columnas_caracteristicas, coeficientes):
                    print(f"{caracteristica}: {coef}")
            y_pred = modelo.predict(X)
            return y, y_pred
        except Exception as e:
            self.logger.error(f"Error en el entrenamiento de regresión lineal: {str(e)}")
            raise ValueError(f"Error en el entrenamiento de regresión lineal: {str(e)}")

    def entrenar_regresion_regularizada(self, config: dict):
        """
        Entrena un modelo de regresión regularizada (Lasso, Ridge, ElasticNet) utilizando los datos preprocesados.
        Args:
            config (dict): Un diccionario que contiene la configuración del modelo de regresión regularizada,
                incluyendo el tipo de regularización (tipo_regularizacion) y el valor de alpha.

        Returns: tuple: Un par de arrays, y_test y y_pred, que contienen los valores reales y predichos en el conjunto
        de prueba.
        """
        try:
            columnas_caracteristicas = ['Rainfall', 'Humidity3pm', 'Cloud3pm']
            variable_objetivo = 'RainfallTomorrow'
            X = self.data_clean[columnas_caracteristicas]
            y = self.data_clean[variable_objetivo]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if config['tipo_regularizacion'] == 'Lasso':
                modelo = Lasso(alpha=config['alpha'])
            elif config['tipo_regularizacion'] == 'Ridge':
                modelo = Ridge(alpha=config['alpha'])
            elif config['tipo_regularizacion'] == 'ElasticNet':
                modelo = ElasticNet(alpha=config['alpha'], l1_ratio=0.5)
            else:
                raise ValueError("Tipo de regularización no válido")
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            return y_test, y_pred
        except Exception as e:
            self.logger.error(f"Error en el entrenamiento de regresión regularizada: {str(e)}")
            raise ValueError(f"Error en el entrenamiento de regresión regularizada: {str(e)}")

    def ejecutar_experimento(self):
        """
        Ejecuta un experimento completo que incluye la limpieza de datos, visualización, preprocesamiento y
        entrenamiento de modelos.
        """
        try:
            self.limpiar_datos()
            self.visualizar_datos()
            self.preprocesar_datos()

            modelos = [
                ("Regresión Lineal", self.entrenar_regresion_lineal, {"normalize": True}),
                ("Regresión Lasso", self.entrenar_regresion_regularizada,
                 {"tipo_regularizacion": "Lasso", "alpha": 0.01}),
                ("Regresión Ridge", self.entrenar_regresion_regularizada,
                 {"tipo_regularizacion": "Ridge", "alpha": 0.01}),
                ("Regresión ElasticNet", self.entrenar_regresion_regularizada, {
                    "tipo_regularizacion": "ElasticNet",
                    "alpha": 0.01
                }),
            ]

            for nombre_modelo, funcion_entrenamiento, *configuracion in modelos:
                if len(configuracion) > 0:
                    y_true, y_pred = funcion_entrenamiento(*configuracion)
                else:
                    y_true, y_pred = funcion_entrenamiento()
                metricas = evaluar_modelo(y_true, y_pred)
                print(f"Métricas del modelo {nombre_modelo}:")
                print(metricas)

        except Exception as e:
            self.logger.error(f"Error en la ejecución del experimento: {str(e)}")
            raise ValueError(f"Error en la ejecución del experimento: {str(e)}")
