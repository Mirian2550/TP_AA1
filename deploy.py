import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, f1_score
import joblib

# Cargar tu conjunto de datos (reemplaza 'data/weatherAUS_clean.csv' con la ruta correcta)
data = pd.read_csv('data/weatherAUS_clean.csv')

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop(['RainTomorrow',  'RainfallTomorrow'], axis=1)
y_regression = data['RainfallTomorrow']
y_classification = data['RainTomorrow']
X_train, X_test, y_train_regression, y_test_regression, y_train_classification, y_test_classification = train_test_split(
    X, y_regression, y_classification, test_size=0.2, random_state=42)

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

# Crear el pipeline completo con el preprocesamiento y el modelo de regresión lineal
pipeline_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Entrenar el pipeline de regresión
pipeline_regression.fit(X_train, y_train_regression)

# Calcular el coeficiente de determinación (R^2)
y_pred_regression = pipeline_regression.predict(X_test)
r2 = r2_score(y_test_regression, y_pred_regression)
print(f"Coeficiente de determinación (R^2) del modelo de regresión: {r2}")

# Crear el pipeline completo con el preprocesamiento y el modelo de regresión logística
pipeline_classification = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# Entrenar el pipeline de clasificación
pipeline_classification.fit(X_train, y_train_classification)

# Calcular el puntaje F1
y_pred_classification = pipeline_classification.predict(X_test)
f1 = f1_score(y_test_classification, y_pred_classification)
print(f"Puntaje F1 del modelo de clasificación: {f1}")

# Guardar los modelos en archivos
joblib.dump(pipeline_regression, 'weather_regression.joblib')
joblib.dump(pipeline_classification, 'weather_classification.joblib')
