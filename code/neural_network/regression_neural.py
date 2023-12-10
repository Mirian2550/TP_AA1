import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna

class RegressionNeuralNetwork:
    def __init__(self, data):
        self.data = data
        self.features = ['MinTemp', 'MaxTemp', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Rainfall']
        self.best_params = self.optimize_hyperparameters()
        self.model = self.build_model()
        self.X_train_normalized = None
        self.X_mean, self.X_std = None, None

    def optimize_hyperparameters(self):
        def objective(trial):
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(1, activation=None, input_shape=(len(self.features),))
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            X_train, X_val, y_train, y_val = train_test_split(
                self.data[self.features], self.data['Rainfall'], test_size=0.2, random_state=42
            )

            # Normalizar los datos
            self.X_mean, self.X_std = X_train.mean(), X_train.std()
            X_train_normalized = (X_train - self.X_mean) / self.X_std
            X_val_normalized = (X_val - self.X_mean) / self.X_std

            history = model.fit(X_train_normalized, y_train, epochs=10, batch_size=32,
                                validation_data=(X_val_normalized, y_val), verbose=0)
            mse = mean_squared_error(y_val, model.predict(X_val_normalized).flatten())

            # Guardar datos normalizados para su uso posterior
            self.X_train_normalized = X_train_normalized

            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        return study.best_params

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation=None, input_shape=(len(self.features),))
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.best_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def regression(self):
        X = self.data[self.features]
        y = self.data['Rainfall']

        if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Invalid data. Please check your input data.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizar los datos
        self.X_mean, self.X_std = X_train.mean(), X_train.std()
        self.X_train_normalized = (X_train - self.X_mean) / self.X_std
        X_test_normalized = (X_test - self.X_mean) / self.X_std

        # Entrenar el modelo con todo el conjunto de entrenamiento
        self.model.fit(self.X_train_normalized, y_train, epochs=10, batch_size=32, verbose=1)

        # Realizar predicciones en el conjunto de prueba
        predictions = self.model.predict(X_test_normalized).flatten()
        mse = mean_squared_error(y_test, predictions)
        print(f"Error Cuadrático Medio en el conjunto de prueba: {mse}")
