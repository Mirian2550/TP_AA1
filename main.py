from ModeloPrediccionLluvia.main import ModeloPrediccionLluvia, evaluar_regresion_logistica
archivo_datos = 'data/weatherAUS.csv'
modelo = ModeloPrediccionLluvia(archivo_datos)
resultados = modelo.regresiones_lineales()
y_test, y_pred = modelo.regresion_logistica()
evaluar_regresion_logistica(y_test=y_test, y_pred=y_pred)

"""


# Crea un gráfico de dispersión para visualizar las predicciones
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', label='Predicciones')
plt.plot([0, 1], [0, 1], 'r--', label='Línea de Referencia')  # Línea de referencia

plt.title('Gráfico de Dispersión de Predicciones vs. Resultados Reales')
plt.xlabel('Resultados Reales')
plt.ylabel('Predicciones')
plt.legend()
plt.grid(True)
plt.show()
"""

