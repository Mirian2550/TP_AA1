import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class RegressionNeuralNetwork:
    def __init__(self, data):
        self.data = data
        self.features = ['MinTemp', 'MaxTemp', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Rainfall']
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(len(self.features),)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def regression(self):
        X = self.data[self.features]
        y = self.data['Rainfall']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        predictions = self.model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, predictions)
        print(f"Error Cuadrático Medio en el conjunto de prueba: {mse}")



