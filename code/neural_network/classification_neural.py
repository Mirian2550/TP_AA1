import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt


class ClassificationNeuralNetwork:
    def __init__(self, data):
        self.data = data
        self.features = [column for column in data.columns if (column != 'RainTomorrow' or column !='RainfallTomorrow') ]

    def build_model(self, trial):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(len(self.features),)))

        for i in range(trial.suggest_int('num_layers', 1, 3)):
            model.add(tf.keras.layers.Dense(
                trial.suggest_int(f'n_units_l{i}', 1, 16),
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01),
            ))
            model.add(tf.keras.layers.Dropout(0.5))  # Puedes ajustar la tasa de dropout
            model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def objective(self, trial):
        X = self.data[self.features]
        y = self.data['RainTomorrow']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.build_model(trial)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        predictions = (model.predict(X_test) > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, predictions)
        return 1.0 - accuracy  # Objetivo es minimizar la métrica, por lo que restamos de 1

    def optimize_hyperparameters(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=50)  # Puedes ajustar el número de trials

        print(f"Mejor valor encontrado: {study.best_value}")
        print(f"Mejores hiperparámetros: {study.best_params}")

        return study.best_params

    def classification(self):
        best_params = self.optimize_hyperparameters()
        model = self.build_model(optuna.trial.FixedTrial(best_params))

        X = self.data[self.features]
        y = self.data['RainTomorrow']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        predictions = (model.predict(X_test) > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print(f"Precisión en el conjunto de prueba: {precision}")
        print(f"Recall en el conjunto de prueba: {recall}")
        print(f"F1-score en el conjunto de prueba: {f1}")
        print(f"Exactitud en el conjunto de prueba: {accuracy}")

    def plot_learning_curves(self):
        X = self.data[self.features]
        y = self.data['RainTomorrow']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.build_model(optuna.trial.FixedTrial(self.optimize_hyperparameters()))

        train_errors, val_errors = [], []
        for m in range(1, len(X_train)):
            model.fit(X_train[:m], y_train[:m], epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_test)
            train_errors.append(1 - accuracy_score(y_train[:m], (y_train_predict > 0.5).astype(int).flatten()))
            val_errors.append(1 - accuracy_score(y_test, (y_val_predict > 0.5).astype(int).flatten()))

        plt.plot(train_errors, "r-+", linewidth=2, label="Conjunto de entrenamiento")
        plt.plot(val_errors, "b-", linewidth=3, label="Conjunto de prueba")
        plt.legend(loc="upper right", fontsize=14)
        plt.xlabel("Tamaño del conjunto de entrenamiento", fontsize=14)
        plt.ylabel("Error", fontsize=14)
        plt.show()