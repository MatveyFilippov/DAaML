import model
from data_processor import test_data


m = model.get_model()
print(f'Mean Absolute Error: {model.get_mae():.2f}')
d = test_data.get_data(rooms=3)
p = m.predict(d)
print(d)
print(p)
