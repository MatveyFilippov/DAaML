import os
import pandas as pd
import settings
import pickle


__COLUMNS_TO_KEEP = None
os.makedirs(settings.PATH_TO_OBJS_DIR, exist_ok=True)


def remove_unused_columns(data: pd.DataFrame) -> pd.DataFrame:
    global __COLUMNS_TO_KEEP
    if __COLUMNS_TO_KEEP is None:
        __COLUMNS_TO_KEEP = settings.FEATURE_COLUMNS.copy()
        __COLUMNS_TO_KEEP.append(settings.TARGET_COLUMN)
    columns_to_keep = list(set(__COLUMNS_TO_KEEP) & set(data.columns))
    return data[columns_to_keep]


class ImportableObject:
    def __init__(self, object_name: str):
        if not object_name.endswith(".pickle"):
            object_name += ".pickle"
        self.__PATH_TO_OBJECT = os.path.join(settings.PATH_TO_OBJS_DIR, object_name)
        self.__OBJ = None

    def reread(self):
        try:
            with open(self.__PATH_TO_OBJECT, "rb") as pf:
                self.__OBJ = pickle.load(pf)
        except FileNotFoundError:
            pass

    def rewrite(self):
        with open(self.__PATH_TO_OBJECT, "wb") as pf:
            pickle.dump(self.__OBJ, pf)

    @property
    def object(self):
        if self.__OBJ is None:
            self.reread()
        return self.__OBJ

    @object.setter
    def object(self, obj):
        self.__OBJ = obj
        self.rewrite()
