import pandas as pd
from sklearn.impute import SimpleImputer
import data_processor
import settings

__INPUTER = data_processor.ImportableObject("inputer")
if __INPUTER.object is None:
    __INPUTER.object = SimpleImputer(strategy='median')


def fit_inputer(data: pd.DataFrame):
    __INPUTER.object.fit(data[settings.FEATURE_COLUMNS])
    __INPUTER.rewrite()


def drop_line_with_null_value(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data.dropna(subset=[column])


def fill_null_values_by_median(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(__INPUTER.object.transform(data[settings.FEATURE_COLUMNS]), columns=settings.FEATURE_COLUMNS)
