from sklearn.model_selection import train_test_split


def split_dataset(dataset):
    features = dataset.drop(columns=['RainfallTomorrow', 'RainTomorrow'])
    regression = dataset['RainfallTomorrow']
    classification = dataset['RainTomorrow']

    x_train, x_test, y_train_regression, y_test_regression, y_train_classification, y_test_classification = train_test_split(
        features, regression, classification, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train_regression, y_test_regression, y_train_classification, y_test_classification