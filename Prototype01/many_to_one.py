from keras.models import Sequential
from keras.layers import Dense, Lambda, Dropout
from keras.layers import LSTM
from keras.layers import GRU
import tensorflow as tf
from matplotlib import pyplot
from tensorflow import keras

from numpy.ma import array

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

DATA_FILE = 'dev_telemetry_volvo.csv'

source_df = read_csv(DATA_FILE, parse_dates=True, index_col=0) \
        .iloc[:60, :] \
        .interpolate(method='linear', axis=0)\
        .drop(columns=['total_odometer', 'gsm_signal', 'sped'])\
        .fillna(0)


source_values = source_df.values
source_values = source_values.astype('float32')

# normalize features
scaled_values = DataFrame(MinMaxScaler(feature_range=(0, 1)).fit_transform(source_values))
del source_values

#df_train, df_test = train_test_split(df, test_size=0.1)

features = len(scaled_values.columns)
rows = len(scaled_values)

#exit(0)
#https://newbedev.com/many-to-one-and-many-to-many-lstm-examples-in-keras

#tf.keras.backend.clear_session()

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(1, features)))
#model.add(Dropout(0.2)) #When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped
model.add(LSTM(units=50, activation='relu', return_sequences=True))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print(model.summary())

x=array(scaled_values.index).reshape(rows, 1, 1)
#print(x)

history = model.fit(
        x=x,
        y=scaled_values.values,
        epochs=2000,
        validation_split=0.2,
        verbose=1,
        batch_size=5)

print(history.params)
print(history.history.keys())

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()
pyplot.show()

#model.add(Lambda(lambda x: x[:, -N:, :])) #Select last N from output#model.add(Dense(1))



