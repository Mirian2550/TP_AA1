import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Cargar tu conjunto de datos (reemplaza 'data/weatherAUS_clean.csv' con la ruta correcta)
data = pd.read_csv('data/weatherAUS_clean.csv')

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop(['RainTomorrow',  'WindGustDir'], axis=1)
y = data['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identificar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(exclude=['number']).columns

# Crear un transformer para imputar variables numéricas con la media
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Crear un transformer para imputar variables categóricas con la moda
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Combinar los transformers en un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Crear el pipeline completo con el preprocesamiento y el modelo de regresión logística
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# Entrenar el pipeline
pipeline.fit(X_train, y_train)

# Guardar el pipeline en un archivo
joblib.dump(pipeline, 'weather.joblib')
