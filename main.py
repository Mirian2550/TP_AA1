from ModeloPrediccionLluvia.main import ModeloPrediccionLluvia
archivo_datos = 'data/weatherAUS.csv'
modelo = ModeloPrediccionLluvia(archivo_datos)
resultados = modelo.ejecutar_experimento()
print(resultados)

