from matplotlib import pyplot
from pandas.plotting import lag_plot

from common import load_data_set
#https://www.machinelearningmastery.ru/autoregression-models-time-series-forecasting-python/
DATA_FILE = 'sparse_telemetry_volvo.csv'

dataset = load_data_set(DATA_FILE, 3000)

#print(dataset)

lag_plot(dataset['predict'], lag=1)
pyplot.show()