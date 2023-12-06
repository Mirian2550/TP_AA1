import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassificationNeuralNetwork:
    def __init__(self, data):
        self.data = data
        self.features = ['MinTemp', 'MaxTemp', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
                         'RainToday', 'RainTomorrow', 'RainfallTomorrow', 'WindGustDir_numerico']
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='relu', input_shape=(len(self.features),)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid para clasificación binaria
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def classification(self):
        X = self.data[self.features]
        y = self.data['RainTomorrow']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        predictions = (self.model.predict(X_test) > 0.5).astype(int).flatten()  # Umbral de 0.5 para clasificación binaria
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print(f"Precisión en el conjunto de prueba: {precision}")
        print(f"Recall en el conjunto de prueba: {recall}")
        print(f"F1-score en el conjunto de prueba: {f1}")
        print(f"Exactitud en el conjunto de prueba: {accuracy}")


