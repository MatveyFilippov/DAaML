PATH_TO_TRAIN_DATA = 'house-pricing-prediction/houses_train_data.csv'
PATH_TO_TEST_DATA = 'house-pricing-prediction/houses_test_data.csv'
PATH_TO_OBJS_DIR = "Objects"

FEATURE_COLUMNS = [
    'Rooms',  # количество комнат в доме
    'Distance',  # расстояние от дома до центра города
    'Bedroom2',  # количество спален в доме
    'Bathroom',  # количество ванных комнат в доме
    'Car',  # количество парковочных мест
    'Landsize',  # размер участка
    'BuildingArea',  # жилая площадь
    'YearBuilt'  # год постройки
]
TARGET_COLUMN = 'Price'  # цена на дом
