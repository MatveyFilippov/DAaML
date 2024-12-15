from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import data_processor
import settings
from data_processor import train_data


# Dataset from https://www.kaggle.com/competitions/house-pricing-prediction/data


__MODEL = data_processor.ImportableObject("main_ml_model")


def get_data_to_train_and_test() -> tuple:
    data = train_data.get_data()

    X = data[settings.FEATURE_COLUMNS]
    y = data[settings.TARGET_COLUMN]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def fit_model():
    X_train, _, y_train, _ = get_data_to_train_and_test()
    model = LinearRegression()
    model.fit(X_train, y_train)
    __MODEL.object = model


def get_model() -> LinearRegression:
    if __MODEL.object is None:
        fit_model()
    return __MODEL.object
