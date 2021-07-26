import pandas
from keras import Input
from matplotlib import pyplot
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import array, number

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.python.saved_model import saved_model

DATA_FILE = 'sparse_telemetry_volvo.csv'

raw = read_csv(DATA_FILE)
        #.iloc[:3000, :]

#values = pandas.to_datetime(raw['fmc_date'])
#values.astype(number)
#raw['timestamp'] = values / 1000

#print(raw.head())
#exit(0)

train_x, test_x = train_test_split(raw['fmc_date'].values, test_size=0.2)
train_y, test_y = train_test_split(raw['predict'].values, test_size=0.2)

train_x = train_x.reshape((len(train_x), 1, 1))
test_x = test_x.reshape((len(test_x), 1, 1))

#return_state=True,
model = Sequential()
model.add(Dense(50, activation='tanh', input_shape=(1, 1)))
model.add(Dense(50, activation='tanh'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(50, activation='tanh'))

# model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(1, 1)))
# model.add(LSTM(50, activation='tanh', return_sequences=True))
# model.add(LSTM(50, activation='tanh', return_sequences=True))
# model.add(LSTM(50, activation='tanh'))

model.add(Dense(1))

model.compile(
    loss='mae',
    optimizer='adam')

#model.load_weights('volvo_lstm2')

history = model.fit(
    x=train_x,
    y=train_y,
    epochs=120,
    batch_size=25,
    validation_data=(test_x, test_y),
    verbose=1,
    shuffle=False)

saved_model.save(model, 'volvo_lstm2')

pyplot.plot(history.history['loss'], label=('train (%.4f)' % history.history['loss'][-1]))
pyplot.plot(history.history['val_loss'], label=('test (%.4f)' % history.history['val_loss'][-1]))

predict_y = model.predict(test_x)
predict_y = predict_y[:, 0]

pyplot.title('mean_squared_error:%s' % (sqrt(mean_squared_error(predict_y, test_y))))

figure, plots = pyplot.subplots(
    nrows=2,
    ncols=1,
    sharex=True)

plots[0].plot(predict_y)
plots[0].set_xlabel("predict_y")

plots[1].plot(test_y)
plots[1].set_xlabel("test_y")

pyplot.show()
