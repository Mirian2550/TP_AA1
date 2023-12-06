from ModeloPrediccionLluvia.main import ModeloPrediccionLluvia, evaluar_regresion_logistica
from ModeloBase.regression import RegresionModelBase
from ModeloBase.clasification import ClasificationModelBase

archivo_datos = 'data/weatherAUS.csv'
modelo = ModeloPrediccionLluvia(archivo_datos)
resultados = modelo.regresiones_lineales()
y_test, y_pred = modelo.regresion_logistica()
evaluar_regresion_logistica(y_test=y_test, y_pred=y_pred)
# modelos base de regresion
# this is model base of regression

print('Modelo Base de Regresion')
modelo_regresion = RegresionModelBase(modelo.data_clean)
modelo_regresion.regresion()
print('Modelo de clasificacion ')
modelo_clasificacion = ClasificationModelBase(modelo.data_clean)
modelo_clasificacion.clasificacion()