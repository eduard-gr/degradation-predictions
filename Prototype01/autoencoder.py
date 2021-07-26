from keras import metrics
import keras
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from numpy import random

from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.saved_model import saved_model

import matplotlib.pyplot as pyplot

DATA_FILE = 'telemetry_volvo.csv'

ENCODER_MODULE = 'volvo_encoded_model_tanh'
DECODER_MODULE = 'volvo_decoder_model_tanh'

#parse_dates=True,
#        .iloc[:3000, :]\
raw = read_csv(DATA_FILE, index_col=0) \
        .interpolate(method='linear', axis=0)\
        .drop(columns=['total_odometer', 'gsm_signal', 'sped'])\
        .fillna(0)

source_values = raw.values
source_values = source_values.astype('float32')

source_values = MinMaxScaler(feature_range=(-1, 1)).fit_transform(source_values)
#del raw

print(source_values.shape)

encoder_input = Input(shape=(40, ))

encoder_layer = Dense(units=30, activation='tanh')(encoder_input)
encoder_layer = Dense(units=20, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=10, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=5, activation='tanh')(encoder_layer)
encoder_layer = Dense(units=1, activation='tanh')(encoder_layer)

encoder = Model(encoder_input, encoder_layer)

decoder_input = Input(shape=(1, ))

decoder_layer = Dense(units=5, activation='tanh')(decoder_input)
decoder_layer = Dense(units=10, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=20, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=30, activation='tanh')(decoder_layer)
decoder_layer = Dense(units=40, activation='tanh')(decoder_layer)

decoder = Model(decoder_input, decoder_layer)

auto_input = Input(shape=(40, ))


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

#saved_model.save(encoder, 'volvo_encoded_model_tanh')
#saved_model.save(decoder, 'volvo_decoder_model_tanh')

#lstm_model = saved_model.load('volvo_model')

#output_array = encoder.predict(test)
#print(output_array)

#output_array = auto_encoder.predict(x_test)
#print(output_array)
