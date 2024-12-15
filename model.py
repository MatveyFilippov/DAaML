from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import data_processor
import settings
from data_processor import train_data


__MODEL = data_processor.ImportableObject("main_ml_model")


def __get_data_to_train() -> tuple:
    data = train_data.get_data()

    X = data[settings.FEATURE_COLUMNS]
    y = data[settings.TARGET_COLUMN]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def fit_model() -> str:
    X_train, X_test, y_train, y_test = __get_data_to_train()
    model = LinearRegression()
    model.fit(X_train, y_train)

    __MODEL.object = model

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return f'Mean Absolute Error: {mae:.2f}'


def get_model() -> LinearRegression:
    if __MODEL.object is None:
        fit_model()
    return __MODEL.object
