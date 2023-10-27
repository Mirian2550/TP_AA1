from ModeloPrediccionLluvia.main import ModeloPrediccionLluvia, evaluar_regresion_logistica
archivo_datos = 'data/weatherAUS.csv'
modelo = ModeloPrediccionLluvia(archivo_datos)
resultados = modelo.regresiones_lineales()
y_test, y_pred = modelo.regresion_logistica()
evaluar_regresion_logistica(y_test=y_test, y_pred=y_pred)
