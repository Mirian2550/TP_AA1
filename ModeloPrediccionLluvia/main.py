import pandas as pd
from scipy.stats import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ModeloPrediccionLluvia:
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.data_clean = None
    def limpiar_datos(self):
        ciudades = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
        datos_filtrados = self.data[self.data['Location'].isin(ciudades)]
        # datos_filtrados.loc[:, 'Date'] = pd.to_datetime(datos_filtrados['Date'])
        datos_filtrados.loc[datos_filtrados['Location'].duplicated(), 'MinTemp'] = datos_filtrados['MinTemp'].ffill()
        datos_filtrados.loc[:,  'MinTemp'] = datos_filtrados.groupby('Location')['MinTemp'].ffill()
        datos_filtrados.loc[datos_filtrados['Location'].duplicated(), 'MaxTemp'] = datos_filtrados['MaxTemp'].ffill()
        datos_filtrados.loc[:, 'MaxTemp'] = datos_filtrados.groupby('Location')['MaxTemp'].ffill()
        datos_filtrados.loc[:, 'Rainfall'] = datos_filtrados['Rainfall'].fillna(0)
        datos_filtrados.loc[:, 'Temp9am'] = datos_filtrados['Temp9am'].ffill()
        datos_filtrados.loc[:, 'Temp3pm'] = datos_filtrados['Temp9am'].ffill()
        datos_filtrados.loc[:, 'Humidity3pm'] = datos_filtrados['Humidity3pm'].ffill()

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

    def preprocesar_datos(self):
        pass

    def entrenar_regresion_lineal(self):
        pass

    def entrenar_regresion_regularizada(self, tipo_regularización):
        pass

    def evaluar_modelo(self, modelo, X_test, y_test, y_pred):
        pass

    def ejecutar_experimento(self):
        self.limpiar_datos()
        self.visualizar_datos()
        pass
