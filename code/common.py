import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def split_dataset(dataset):
    scaler = StandardScaler()
    features = dataset.drop(columns=['RainfallTomorrow','RainTomorrow','Date','WindGustDir','WindDir9am','WindDir3pm'])
    regression = dataset['RainfallTomorrow']
    classification = dataset['RainTomorrow']

    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform(features)

    # Convertir los arrays de NumPy en DataFrames de Pandas
    features_imputed_df = pd.DataFrame(features_imputed, columns=features.columns)
    regression_df = pd.DataFrame(regression, columns=['RainfallTomorrow'])
    classification_df = pd.DataFrame(classification, columns=['RainTomorrow'])

    x_train, x_test, y_train_regression, y_test_regression, y_train_classification, y_test_classification = train_test_split(
        features_imputed_df, regression_df, classification_df, test_size=0.2, random_state=42
    )

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Convertir los arrays de NumPy en DataFrames de Pandas
    x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=features.columns)
    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=features.columns)


    return x_train_scaled_df, x_test_scaled_df, y_train_regression, y_test_regression, y_train_classification, y_test_classification

#%%
