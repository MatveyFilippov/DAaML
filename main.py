import model
from data_processor import test_data


m = model.get_model()
d = test_data.get_data(rooms=3)
p = m.predict(d)
print(d)
print(p)
