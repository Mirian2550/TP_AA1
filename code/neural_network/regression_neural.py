import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

class RegressionNeuralNetwork:
    def __init__(self):
        self.model = None
        self.X_train_normalized = None
        self.X_mean, self.X_std = None, None

    def optimize_hyperparameters(self, X_train, y_train):
        def objective(trial):
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(1, activation=None, input_shape=(X_train.shape[1],))
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            # Normalizar los datos
            self.X_mean, self.X_std = X_train.mean(), X_train.std()
            X_train_normalized = (X_train - self.X_mean) / self.X_std

            history = model.fit(X_train_normalized, y_train, epochs=10, batch_size=32, verbose=0)
            mse = mean_squared_error(y_train, model.predict(X_train_normalized).flatten())

            # Guardar datos normalizados para su uso posterior
            self.X_train_normalized = X_train_normalized

            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        return study.best_params

    def build_model(self, best_params):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation=None, input_shape=(self.X_train_normalized.shape[1],))
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def regression(self, X_train, X_test, y_train, y_test):
        if X_train is None or y_train is None or X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Invalid training data. Please check your input data.")
        if X_test is None or y_test is None or X_test.shape[0] == 0 or y_test.shape[0] == 0:
            raise ValueError("Invalid test data. Please check your input data.")

        # Optimizar hiperparámetros
        best_params = self.optimize_hyperparameters(X_train, y_train)

        # Construir el modelo
        self.model = self.build_model(best_params)

        # Normalizar los datos de entrenamiento y prueba
        self.X_mean, self.X_std = X_train.mean(), X_train.std()
        self.X_train_normalized = (X_train - self.X_mean) / self.X_std
        X_test_normalized = (X_test - self.X_mean) / self.X_std

        # Entrenar el modelo con todo el conjunto de entrenamiento
        self.model.fit(self.X_train_normalized, y_train, epochs=10, batch_size=32, verbose=1)

        # Realizar predicciones en el conjunto de prueba
        predictions = self.model.predict(X_test_normalized).flatten()
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Mean Squared Error on test set: {mse}")
        print(f"Mean Absolute Error on test set: {mae}")
        print(f"R-squared score on test set: {r2}")
