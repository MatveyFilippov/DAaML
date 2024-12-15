import pandas as pd
import data_processor
import settings
from data_processor import null_processor


__DATA = data_processor.ImportableObject("train_data")


def get_data() -> pd.DataFrame:
    global __DATA
    if __DATA.object is None:
        reset_data()
    return __DATA.object


def reset_data(only_if_null=False):
    global __DATA

    if only_if_null and isinstance(__DATA.object, pd.DataFrame):
        return

    data = pd.read_csv(settings.PATH_TO_TRAIN_DATA)
    data = data_processor.remove_unused_columns(data)
    data = null_processor.drop_line_with_null_value(data, settings.TARGET_COLUMN)

    null_processor.fit_inputer(data)

    data[settings.FEATURE_COLUMNS] = null_processor.fill_null_values_by_median(data[settings.FEATURE_COLUMNS])

    __DATA.object = data
