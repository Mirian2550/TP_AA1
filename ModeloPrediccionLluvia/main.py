import numpy as np
import pandas as pd
from matplotlib.widgets import Lasso
from scipy.stats import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


class ModeloPrediccionLluvia:
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.data_clean = None
    def limpiar_datos(self):
        ciudades = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
        datos_filtrados = self.data[self.data['Location'].isin(ciudades)]
        # datos_filtrados.loc[:, 'Date'] = pd.to_datetime(datos_filtrados['Date'])
        datos_filtrados.loc[:,  'MinTemp'] = datos_filtrados.groupby('Location')['MinTemp'].ffill()
        datos_filtrados.loc[:, 'MaxTemp'] = datos_filtrados.groupby('Location')['MaxTemp'].ffill()
        datos_filtrados.loc[:, 'Rainfall'] = datos_filtrados['Rainfall'].fillna(0)
        datos_filtrados.loc[:, 'Temp9am'] = datos_filtrados['Temp9am'].ffill()
        datos_filtrados.loc[:, 'Temp3pm'] = datos_filtrados['Temp9am'].ffill()
        datos_filtrados.loc[:, 'Humidity3pm'] = datos_filtrados['Humidity3pm'].ffill()
        datos_filtrados.loc[:, 'Cloud3pm'] = datos_filtrados['Cloud3pm'].ffill()
        umbral_z = 3
        columnas_numericas = datos_filtrados.select_dtypes(include='number').columns
        for columna in columnas_numericas:
            z_scores = stats.zscore(datos_filtrados[columna])
            valores_atipicos = (z_scores > umbral_z) | (z_scores < -umbral_z)
            print(f"Valores atípicos en la columna {columna}:")
            print(datos_filtrados[valores_atipicos])
        self.data_clean = datos_filtrados
        """
        datos_filtrados['Date'] = pd.to_datetime(datos_filtrados['Date'])
        datos_filtrados.sort_values(['Location', 'Date'], inplace=True)

        # Agrupa los datos por ubicación y rellena los valores nulos en 'MinTemp' por la ubicación
        datos_filtrados['MinTemp'] = datos_filtrados.groupby('Location')['MinTemp'].ffill()
        datos_filtrados['MinTemp'].fillna(method='ffill', inplace=True)
        """
        # print(datos_filtrados['MinTemp'])

        """
        total_datos = datos_filtrados.size

        # Calcula la cantidad de datos nulos y el porcentaje de datos nulos por columna
        datos_nulos_por_columna = datos_filtrados.isnull().sum()
        porcentaje_nulos_por_columna = (datos_nulos_por_columna / datos_filtrados.shape[0]) * 100

        # Itera a través de las columnas para mostrar los resultados
        for columna in datos_filtrados.columns:
            print(f"Columna: {columna}")
            print(f"Cantidad de datos nulos: {datos_nulos_por_columna[columna]}")
            print(f"Porcentaje de datos nulos: {porcentaje_nulos_por_columna[columna]:.2f}%")
            print()

        # Imprime la cantidad total de datos y porcentaje de datos nulos en todo el conjunto de datos
        print(f"Cantidad total de datos: {total_datos}")
        print(f"Porcentaje de datos nulos en todo el conjunto de datos: {porcentaje_nulos_por_columna.mean():.2f}%")
        """
    def visualizar_datos(self):
        columns_tmp = [ 'MinTemp', 'MaxTemp', 'Rainfall',
       'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
       'Temp9am', 'Temp3pm',  'RainfallTomorrow']
        """
        for columna in self.data_clean.columns:
            try:
                self.data_clean[columna] = self.data_clean[columna].astype(float)
            except ValueError:
                print(f"Error en la columna {columna}")
        """
        matriz_correlacion = self.data_clean[columns_tmp].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Matriz de Correlación (sin Date)')
        plt.show()
        conteo_clases = self.data_clean['RainTomorrow'].value_counts()

        # Calcula el equilibrio
        porcentaje_clase_positiva = (conteo_clases['Yes'] / len(self.data_clean)) * 100
        porcentaje_clase_negativa = (conteo_clases['No'] / len(self.data_clean)) * 100

        print(f'Porcentaje de clase "Yes" (Lloverá): {porcentaje_clase_positiva:.2f}%')
        print(f'Porcentaje de clase "No" (No lloverá): {porcentaje_clase_negativa:.2f}%')

    def preprocesar_datos(self):
        pass


    def entrenar_regresion_lineal(self):
        # Selecciona las columnas de características y la variable objetivo
        columnas_caracteristicas = ['Rainfall', 'Humidity3pm', 'Cloud3pm']
        variable_objetivo = 'RainfallTomorrow'

        # Divide los datos en conjuntos de entrenamiento y prueba
        X = self.data_clean[columnas_caracteristicas]
        y = self.data_clean[variable_objetivo]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Imputa valores nulos en la variable objetivo (y_train) utilizando la media
        imputer = SimpleImputer(strategy='mean')
        y_train = imputer.fit_transform(y_train.values.reshape(-1, 1))

        # Crea un modelo de regresión lineal
        modelo = LinearRegression()

        # Entrena el modelo
        modelo.fit(X_train, y_train)

        # Realiza predicciones en el conjunto de prueba
        y_pred = modelo.predict(X_test)

        return y_test, y_pred

    def entrenar_regresion_regularizada(self, tipo_regularizacion, alpha=1.0):
        # Selecciona las columnas de características y la variable objetivo
        columnas_caracteristicas = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
                                    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                                    'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
        variable_objetivo = 'RainfallTomorrow'

        # Divide los datos en conjuntos de entrenamiento y prueba
        X = self.data_clean[columnas_caracteristicas]
        y = self.data_clean[variable_objetivo]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if tipo_regularizacion == 'Lasso':
            # Crea un modelo de regresión Lasso
            modelo = Lasso(alpha=alpha)
        elif tipo_regularizacion == 'Ridge':
            # Crea un modelo de regresión Ridge
            modelo = Ridge(alpha=alpha)
        elif tipo_regularizacion == 'ElasticNet':
            # Crea un modelo de regresión Elastic Net
            modelo = ElasticNet(alpha=alpha, l1_ratio=0.5)
        else:
            raise ValueError("Tipo de regularización no válido")

        # Entrena el modelo
        modelo.fit(X_train, y_train)

        # Realiza predicciones en el conjunto de prueba
        y_pred = modelo.predict(X_test)

        return y_test, y_pred

    def evaluar_modelo(self, y_true, y_pred):
        # Imputa valores nulos en y_true y y_pred utilizando la media
        imputer = SimpleImputer(strategy='mean')
        y_true = y_true.values.reshape(-1, 1)
        y_pred = y_pred.values.reshape(-1, 1)

        # Calcula métricas de rendimiento
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

    def ejecutar_experimento(self):
        self.limpiar_datos()

        # Visualiza los datos
        self.visualizar_datos()

        # Preprocesa los datos si es necesario
        self.preprocesar_datos()

        # Entrenamiento y evaluación de modelos
        # Entrenar y evaluar el modelo de regresión lineal
        y_true, y_pred = self.entrenar_regresion_lineal()
        metricas = self.evaluar_modelo(y_true, y_pred)
        print("Métricas del modelo de regresión lineal:")
        print(metricas)

        # Entrenar y evaluar el modelo de regresión regularizada (Lasso)
        y_true, y_pred = self.entrenar_regresion_regularizada("Lasso", alpha=0.01)
        metricas = self.evaluar_modelo(y_true, y_pred)
        print("Métricas del modelo de regresión Lasso:")
        print(metricas)

        # Entrenar y evaluar el modelo de regresión regularizada (Ridge)
        y_true, y_pred = self.entrenar_regresion_regularizada("Ridge", alpha=0.01)
        metricas = self.evaluar_modelo(y_true, y_pred)
        print("Métricas del modelo de regresión Ridge:")
        print(metricas)

        # Entrenar y evaluar el modelo de regresión regularizada (ElasticNet)
        y_true, y_pred = self.entrenar_regresion_regularizada("ElasticNet", alpha=0.01)
        metricas = self.evaluar_modelo(y_true, y_pred)
        print("Métricas del modelo de regresión ElasticNet:")
        print(metricas)

