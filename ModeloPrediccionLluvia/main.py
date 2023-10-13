import pandas as pd
class ModeloPrediccionLluvia:
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.data_clean = None
    def limpiar_datos(self):
        ciudades = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']
        datos_filtrados = self.data[self.data['Location'].isin(ciudades)]
        print(datos_filtrados)

    def visualizar_datos(self):
        pass

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
        pass
