from keras import metrics
import keras
import tensorflow as tf
import os

from scipy.stats import stats
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from numpy import random

from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.saved_model import saved_model

import matplotlib.pyplot as pyplot

DATA_FILE = 'telemetry_volvo.csv'

ENCODER_MODULE = 'volvo_encoded_model_tanh_16'
DECODER_MODULE = 'volvo_decoder_model_tanh_16'

#parse_dates=True,
#        .iloc[:3000, :]\
raw = read_csv(DATA_FILE, index_col=0) \
        .drop(columns=[
            'total_odometer',
            'gsm_signal',
            'number_of_dtc',
            'vehicle_speed',
            'runtime_since_engine_start',
            'distance_traveled_mil_on',
            'fuel_level',
            'distance_since_codes_clear',
            'absolute_load_value',
            'time_since_codes_cleared',
            'absolute_fuel_rail_pressure',
            'engine_oil_temperature',
            'fuel_injection_timing',
            'fuel_rate',
            'battery_current',
            'gnss_status',
            'data_mode',
            'gnss_pdop',
            'gnss_hdop',
            'sleep_mode',
            'ignition',
            'movement',
            'active_gsm_operator',
            'green_driving_type',
            'unplug',
            'green_driving_value',
            'sped']) \
        .interpolate(method='linear', axis=0)\
        .fillna(0)

source_values = raw.values
source_values = source_values.astype('float32')

#Primenit metod zscoringa https://www.statology.org/z-score-python/
#source_values = MinMaxScaler(feature_range=(-1, 1)).fit_transform(source_values)
source_values = stats.zscore(source_values, axis=1)
#del raw

print(source_values.shape)
source_values.shape[1]
encoder_input = Input(shape=(source_values.shape[1], ))

encoder_layer = Dense(units=15, activation='tanh')(encoder_input)
encoder_layer = Dense(units=14, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=13, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=12, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=11, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=10, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=9, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=8, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=7, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=6, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=5, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=4, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=3, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=2, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=1, activation='tanh')(encoder_layer)

encoder = Model(encoder_input, encoder_layer)

decoder_input = Input(shape=(1, ))

decoder_layer = Dense(units=2, activation='tanh')(decoder_input)
decoder_layer = Dense(units=3, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=4, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=5, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=6, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=7, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=8, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=9, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=10, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=11, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=12, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=13, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=14, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=15, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=16, activation='tanh')(decoder_layer)

decoder = Model(decoder_input, decoder_layer)

auto_input = Input(shape=(source_values.shape[1], ))


encoded = encoder(auto_input)
decoded = decoder(encoded)

auto_encoder = Model(auto_input, decoded)
#auto_encoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

auto_encoder.summary()


encoder.load_weights(ENCODER_MODULE)
decoder.load_weights(DECODER_MODULE)


result = DataFrame(
    encoder.predict(source_values),
    columns=['predict'],
    index=raw.index)

result.to_csv('sparse_telemetry_volvo.csv', index=True, header=True)

#result['engine_rpm'] = raw['engine_rpm']
#result['speed'] = raw['speed']


# figure, plots = pyplot.subplots(
#     nrows=len(result.columns),
#     ncols=1,
#     sharex=True)
#
# index=0
# for column in result:
#     canvas = plots[index]
#     canvas.plot(result[column].values)
#     canvas.set_xlabel(column)
#     index = index + 1
#
# pyplot.show()

#print(result_df.head())

# train, test = train_test_split(source_values, test_size=0.2)
#
# history = auto_encoder.fit(
#     x=train,
#     y=train,
#     validation_data=(test, test),
#     epochs=150,
#     batch_size=20)
#
# pyplot.plot(history.history['loss'], label=('train (%.4f)' % history.history['loss'][-1]))
# pyplot.plot(history.history['val_loss'], label=('test (%.4f)' % history.history['val_loss'][-1]))
# pyplot.show()

# saved_model.save(encoder, ENCODER_MODULE)
# saved_model.save(decoder, DECODER_MODULE)

