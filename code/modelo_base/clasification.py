from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve
import pandas as pd
import matplotlib.pyplot as plt
class ClasificacionModelBase:
    def __init__(self):
        """
        Inicializa la instancia del modelo de clasificación.
        """
        self.model = 'LogisticRegression Simple'

    def logistic(self, _x_train, _x_test, _y_train_classification, _y_test_classification):
        try:
            modelo = LogisticRegression(max_iter=10000)
            modelo.fit(_x_train, _y_train_classification)
            y_pred = modelo.predict(_x_test)
            return _x_test, _y_test_classification, y_pred, modelo
        except Exception as e:
            raise ValueError(f"Error en el entrenamiento de regresión logística: {str(e)}")

    def logic_metrics(self,y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Métricas
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print(f'ROC-AUC: {roc_auc:.2f}')

        # Matriz de confusión
        print("Matiz de confusión:")
        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                           columns=["pred: No", "Pred: Si"],
                           index=["Real: No", "Real: si"]))

        # Calculo la ROC y el AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        # Grafico la curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.show()