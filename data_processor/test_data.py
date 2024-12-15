import pandas as pd
import data_processor
import settings
from data_processor import null_processor
from data_processor import train_data


__REQUIRED_KEYS_LOWER = {k.replace("_", "").lower(): k for k in settings.FEATURE_COLUMNS}
__NULL_DATA = {k: None for k in settings.FEATURE_COLUMNS}


def get_random_data() -> pd.DataFrame:
    train_data.reset_data(only_if_null=True)
    data = pd.read_csv(settings.PATH_TO_TEST_DATA).sample(n=1)
    data = data_processor.remove_unused_columns(data)
    return null_processor.fill_null_values_by_median(data)


def get_data(**kwargs) -> pd.DataFrame:
    data = __NULL_DATA.copy()
    for k, v in kwargs.items():
        k = k.replace("_", "").lower()
        if k in __REQUIRED_KEYS_LOWER:
            data[__REQUIRED_KEYS_LOWER[k]] = v

    train_data.reset_data(only_if_null=True)
    data = pd.DataFrame([data])
    data = data_processor.remove_unused_columns(data)
    return null_processor.fill_null_values_by_median(data)
