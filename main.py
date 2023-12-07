from ModeloPrediccionLluvia.main import ModeloPrediccionLluvia, evaluar_regresion_logistica
from ModeloBase.regression import RegresionModelBase
from ModeloBase.clasification import ClasificationModelBase
from NeuralNetwork.classification_neural import ClassificationNeuralNetwork
from NeuralNetwork.regression_neural import RegressionNeuralNetwork
from shap_analyzer.shap_analyzer import SHAPAnalyzer

archivo_datos = 'data/weatherAUS.csv'
modelo = ModeloPrediccionLluvia(archivo_datos)
modelos = modelo.regresiones_lineales()
for _model in modelos:
    trained_model = _model.get('modelo')
    analyzer = SHAPAnalyzer(trained_model, _model.get('x_test'))
    analyzer.summary_plot()
    #analyzer = SHAPAnalyzer(_model.get('modelo'), _model.get('x_test'))
    #analyzer.summary_plot()
    analyzer.force_plot(index=0)
    break
#y_test, y_pred = modelo.regresion_logistica()
#evaluar_regresion_logistica(y_test=y_test, y_pred=y_pred)
"""
print('====================Modelo Base de Regresion=================')
modelo_regresion = RegresionModelBase(modelo.data_clean)
modelo_regresion.regresion()

print('=============================Modelo de clasificacion=========================================================')
modelo_clasificacion = ClasificationModelBase(modelo.data_clean)
modelo_clasificacion.clasificacion()
print('=============================clasificacion con redes neuronales================================================')
classification_nn_model = ClassificationNeuralNetwork(modelo.data_clean)
trained_model = classification_nn_model.classification()
print('=============================regresion con redes neuronales====================================================')
regression_nn_model = RegressionNeuralNetwork(modelo.data_clean)
# regression_nn_model.regression()
regression_nn_model.regression_with_shap()
"""
