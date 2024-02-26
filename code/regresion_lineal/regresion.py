import logging

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, roc_curve
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight


class RegressionLineal:
    def __init__(self, data=None):
        self.data = data
        self.logger = logging.getLogger("regression")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

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

    def logistic(self):
        """
        Entrena un modelo de regresión logística utilizando los datos preprocesados.

        Returns:
            tuple: Una tupla que contiene x_test, y_test, y_pred y el modelo entrenado.
        """
        try:
            x = self.data[['Humidity3pm', 'Cloud3pm', 'Rainfall']]

            y = self.data['RainTomorrow']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            modelo = LogisticRegression(max_iter=10000)
            modelo.fit(x_train, y_train)
            y_pred = modelo.predict(x_test)
            return x_test, y_test, y_pred, modelo
        except Exception as e:
            self.logger.error(f"Error en el entrenamiento de regresión logística: {str(e)}")
            raise ValueError(f"Error en el entrenamiento de regresión logística: {str(e)}")

    def logistic_balanced(self):
        try:
            x = self.data[['Humidity3pm', 'Cloud3pm', 'Rainfall']]
            y = self.data['RainTomorrow']

            # Calcular pesos de clases para balanceo
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            modelo = LogisticRegression(max_iter=10000, class_weight=dict(zip(np.unique(y), class_weights)))
            modelo.fit(x_train, y_train)
            y_pred = modelo.predict(x_test)

            return x_test, y_test, y_pred, modelo

        except Exception as e:
            self.logger.error(f"Error en el entrenamiento de regresión logística: {str(e)}")
            raise ValueError(f"Error en el entrenamiento de regresión logística: {str(e)}")

    def optimize_hyperparameters_logistic(self, param_grid, cv=5, n_iter=10):
        try:
            x = self.data[['Humidity3pm', 'Cloud3pm', 'Rainfall']]
            y = self.data['RainTomorrow']

            # Dividir los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Normalizar características si es necesario
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Crear modelo de regresión logística
            modelo = LogisticRegression(max_iter=10000)

            # Configurar la búsqueda aleatoria de hiperparámetros
            random_search = RandomizedSearchCV(modelo, param_distributions=param_grid, n_iter=n_iter, cv=cv)

            # Realizar la búsqueda aleatoria
            random_search.fit(x_train, y_train)

            # Obtener el mejor modelo
            best_model = random_search.best_estimator_

            # Realizar predicciones en el conjunto de prueba
            y_pred = best_model.predict(x_test)

            return x_test, y_test, y_pred, best_model

        except Exception as e:
            print(f"Error en la función optimize_hyperparameters_logistic: {str(e)}")
            return None
    """
    def classic(self):
        try:
            columnas_caracteristicas = [
                'Rainfall', 'Humidity3pm','Cloud3pm'
            ]
            x = self.data[columnas_caracteristicas]
            y = self.data['RainfallTomorrow']

            # Dividir los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            modelo = LinearRegression()
            # Entrenar el modelo
            modelo.fit(x_train, y_train)
            # Realizar predicciones en el conjunto de prueba
            y_pred = modelo.predict(x_test)
            return x_test, y_test, y_pred, modelo

        except Exception as e:
            print(f"Error en la función classic: {str(e)}")
            return None

    """
    def classic(self, x_train, x_test, y_train_regression, y_test_regression):
        try:
            modelo = LinearRegression()
            print("nulitos en x_train",x_train.isna().sum())
            print("nulitos en x_test",x_test.isna().sum())
            print("nulitos en y_train_regression", y_train_regression.isna().sum())
            print("nulitos en y_test_regression", y_test_regression.isna().sum())
            # Entrenar el modelo
            modelo.fit(x_train[['Humidity3pm', 'Cloud3pm', 'Rainfall']], y_train_regression)
            # Realizar predicciones en el conjunto de prueba
            y_pred = modelo.predict(x_test)
            print(f"El modelo",y_pred)
            return y_test_regression, y_pred, modelo

        except Exception as e:
            print(f"Error en la función classic: {str(e)}")
            raise ValueError("Error en la función classic")


    def cross_validate(self, x_test, y_test, modelo, cv=5):
        try:
            # Realizar validación cruzada y obtener métricas
            y_pred_cv = cross_val_predict(modelo, x_test, y_test, cv=cv)

            mse = mean_squared_error(y_test, y_pred_cv)
            r2 = r2_score(y_test, y_pred_cv)
            mae = mean_absolute_error(y_test, y_pred_cv)

            # Imprimir las métricas
            print(f'Mean Squared Error (CV): {mse}')
            print(f'R^2 Score (CV): {r2}')
            print(f'Mean Absolute Error (CV): {mae}')

            return mse, r2, mae

        except Exception as e:
            print(f"Error en la función cross_validate: {str(e)}")
            return None

    def gradient_descent(self, learning_rate=0.01, num_iterations=1000):
        try:
            columnas_caracteristicas = [
                'Rainfall', 'Humidity3pm','Cloud3pm'
            ]
            variable_objetivo = 'RainfallTomorrow'
            x = self.data[columnas_caracteristicas]
            y = self.data[variable_objetivo]

            # Dividir los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Crear modelo de regresión lineal con descenso de gradiente estocástico
            modelo = SGDRegressor(learning_rate='constant', eta0=learning_rate, max_iter=num_iterations)

            # Entrenar el modelo
            modelo.fit(x_train, y_train)

            # Realizar predicciones en el conjunto de prueba
            y_pred = modelo.predict(x_test)

            return x_test, y_test, y_pred, modelo

        except Exception as e:
            print(f"Error en la función gradient_descent: {str(e)}")
            return None

    def gradient_descent_optimize_hyperparameters(self, param_grid, cv=5, n_iter=10):
        try:
            columnas_caracteristicas = [
                'Rainfall', 'Humidity3pm','Cloud3pm'
            ]
            variable_objetivo = 'RainfallTomorrow'
            x = self.data[columnas_caracteristicas]
            y = self.data[variable_objetivo]

            # Dividir los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Normalizar características
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Inicializar el modelo
            modelo = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000)

            # Configurar la búsqueda aleatoria de hiperparámetros
            random_search = RandomizedSearchCV(modelo, param_distributions=param_grid, n_iter=n_iter, cv=cv)

            # Realizar la búsqueda aleatoria
            random_search.fit(x_train, y_train)

            # Obtener el mejor modelo
            best_model = random_search.best_estimator_

            # Realizar predicciones en el conjunto de prueba
            y_pred = best_model.predict(x_test)

            return x_test, y_test, y_pred, best_model

        except Exception as e:
            print(f"Error en la función optimize_hyperparameters: {str(e)}")
            return None

    def ridge_regression(self, alpha=1.0):
        try:
            columnas_caracteristicas = [
                'Rainfall', 'Humidity3pm','Cloud3pm'
            ]
            variable_objetivo = 'RainfallTomorrow'
            x = self.data[columnas_caracteristicas]
            y = self.data[variable_objetivo]

            # Dividir los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Crear una instancia de Ridge Regression
            ridge_model = Ridge(alpha=alpha)

            # Entrenar el modelo
            ridge_model.fit(x_train, y_train)

            # Realizar predicciones en el conjunto de prueba
            y_pred = ridge_model.predict(x_test)

            return x_test, y_test, y_pred, ridge_model

        except Exception as e:
            print(f"Error en la función ridge_regression: {str(e)}")
            return None

    def lasso_regression(self, alpha=1.0):
        try:
            columnas_caracteristicas = [
                'Rainfall', 'Humidity3pm','Cloud3pm'
            ]
            variable_objetivo = 'RainfallTomorrow'
            x = self.data[columnas_caracteristicas]
            y = self.data[variable_objetivo]

            # Dividir los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Crear una instancia de Lasso Regression
            lasso_model = Lasso(alpha=alpha)

            # Entrenar el modelo
            lasso_model.fit(x_train, y_train)

            # Realizar predicciones en el conjunto de prueba
            y_pred = lasso_model.predict(x_test)

            return x_test, y_test, y_pred, lasso_model

        except Exception as e:
            print(f"Error en la función lasso_regression: {str(e)}")
            return None

    def elasticnet_regression(self, alpha=1.0, l1_ratio=0.5):
        try:
            columnas_caracteristicas = [
                'Rainfall', 'Humidity3pm','Cloud3pm'
            ]
            variable_objetivo = 'RainfallTomorrow'
            x = self.data[columnas_caracteristicas]
            y = self.data[variable_objetivo]

            # Dividir los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Crear una instancia de ElasticNet Regression
            elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

            # Entrenar el modelo
            elasticnet_model.fit(x_train, y_train)

            # Realizar predicciones en el conjunto de prueba
            y_pred = elasticnet_model.predict(x_test)

            return x_test, y_test, y_pred, elasticnet_model

        except Exception as e:
            print(f"Error en la función elasticnet_regression: {str(e)}")
            return None

    def optimize_hyperparameters(self, model_name, param_grid):
        if model_name == 'Lasso':
            model = Lasso()
        elif model_name == 'Ridge':
            model = Ridge()
        elif model_name == 'ElasticNet':
            model = ElasticNet()
        else:
            print(f"Modelo no válido: {model_name}")
            return None

        try:
            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                self.data[[
                'Rainfall', 'Humidity3pm','Cloud3pm'
            ]], self.data['RainfallTomorrow'], test_size=0.2, random_state=42
            )

            x_test, y_test, y_pred, best_model = self._optimize_hyperparameters(
                model, param_grid, X_train, y_train, X_test, y_test
            )
            return x_test, y_test, y_pred, best_model
        except Exception as e:
            print(f"Error en optimize_hyperparameters: {str(e)}")
            return None

    def _optimize_hyperparameters(self, model, param_grid, X_train, y_train, X_test, y_test):
        # ...

        # Ajustar el modelo con búsqueda aleatoria
        random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5)
        random_search.fit(X_train, y_train)

        # Obtener los mejores hiperparámetros y el mejor modelo
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        # Realizar predicciones en el conjunto de prueba
        y_pred = best_model.predict(X_test)

        return X_test, y_test, y_pred, best_model

    def metrics(self, y_true, y_pred):
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
