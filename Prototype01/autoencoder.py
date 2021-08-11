from math import sqrt

from keras import metrics
import keras
import tensorflow as tf
import os

from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from numpy import random

from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.saved_model import saved_model

import matplotlib.pyplot as pyplot

from common import load_data_set

DATA_FILE = 'telemetry_volvo.csv'
RESULT_FILE = 'sparse_telemetry_volvo.csv'

AUTOENCODER_MODULE = 'volvo_autoencoder_model_tanh_16'
ENCODER_MODULE = 'volvo_encoded_model_tanh_16'
DECODER_MODULE = 'volvo_decoder_model_tanh_16'

def display_result(source, predicted):
    figure, plots = pyplot.subplots(
        nrows=source.shape[1] + 1,
        ncols=1,
        sharex=True,
        figsize=(24, 192))

    pyplot.subplots_adjust(hspace=0.5, bottom=0.05, top=0.95)
    index = 1

    canvas = plots[0]
    canvas.plot(predicted)
    canvas.set_xlabel("Predicted")

    for column in source:
        canvas = plots[index]
        canvas.plot(source[column].values)
        canvas.set_xlabel(column)
        index = index + 1

    pdf = PdfPages('autoencoder.pdf')
    pdf.savefig()
    pdf.close()

def save_to_csv(predicted, name):
    result = DataFrame(
        predicted,
        columns=['predict'])

    result.to_csv(name, index=True, header=True)

source = load_data_set(DATA_FILE)


source_values = source.values
source_values = source_values.astype('float32')

#Primenit metod zscoringa https://www.statology.org/z-score-python/
#source_values = MinMaxScaler(feature_range=(-1, 1)).fit_transform(source_values)
ranked_values = stats.zscore(source_values, axis=1)


encoder_input = Input(shape=(ranked_values.shape[1], ))
#selu
activation = 'relu'
encoder_layer = Dense(units=15, activation=activation)(encoder_input)
#encoder_layer = Dense(units=14, activation=activation)(encoder_layer)
#encoder_layer = Dense(units=13, activation=activation)(encoder_layer)
#encoder_layer = Dense(units=12, activation=activation)(encoder_layer)
#encoder_layer = Dense(units=11, activation=activation)(encoder_layer)
encoder_layer = Dense(units=10, activation=activation)(encoder_layer)
#encoder_layer = Dense(units=9, activation=activation)(encoder_layer)
#encoder_layer = Dense(units=8, activation=activation)(encoder_layer)
#encoder_layer = Dense(units=7, activation=activation)(encoder_layer)
#encoder_layer = Dense(units=6, activation=activation)(encoder_layer)
encoder_layer = Dense(units=5, activation=activation)(encoder_layer)
#encoder_layer = Dense(units=4, activation=activation)(encoder_layer)
#encoder_layer = Dense(units=3, activation=activation)(encoder_layer)
encoder_layer = Dense(units=2, activation=activation)(encoder_layer)
encoder_layer = Dense(units=1, activation=activation)(encoder_layer)

encoder = Model(encoder_input, encoder_layer)

decoder_input = Input(shape=(1, ))

decoder_layer = Dense(units=2, activation=activation)(decoder_input)
#decoder_layer = Dense(units=3, activation=activation)(decoder_layer)
#decoder_layer = Dense(units=4, activation=activation)(decoder_layer)
decoder_layer = Dense(units=5, activation=activation)(decoder_layer)
#decoder_layer = Dense(units=6, activation=activation)(decoder_layer)
#decoder_layer = Dense(units=7, activation=activation)(decoder_layer)
#decoder_layer = Dense(units=8, activation=activation)(decoder_layer)
#decoder_layer = Dense(units=9, activation=activation)(decoder_layer)
decoder_layer = Dense(units=10, activation=activation)(decoder_layer)
#decoder_layer = Dense(units=11, activation=activation)(decoder_layer)
#decoder_layer = Dense(units=12, activation=activation)(decoder_layer)
#decoder_layer = Dense(units=13, activation=activation)(decoder_layer)
#decoder_layer = Dense(units=14, activation=activation)(decoder_layer)
decoder_layer = Dense(units=15, activation=activation)(decoder_layer)
decoder_layer = Dense(units=16, activation=activation)(decoder_layer)

decoder = Model(decoder_input, decoder_layer)

auto_input = Input(shape=(ranked_values.shape[1], ))

encoded = encoder(auto_input)
decoded = decoder(encoded)
auto_encoder = Model(auto_input, decoded)

auto_encoder.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
#optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
#auto_encoder.compile(loss='binary_crossentropy', optimizer='adadelta' )

auto_encoder.summary()

# train, test = train_test_split(
#     ranked_values,
#     test_size=0.2)
#
# history = auto_encoder.fit(
#     x=train,
#     y=train,
#     validation_data=(test, test),
#     epochs=150,
#     batch_size=20)
#
# pyplot.figure()
# pyplot.plot(history.history['loss'], label=('train (%.4f)' % history.history['loss'][-1]))
# pyplot.plot(history.history['val_loss'], label=('test (%.4f)' % history.history['val_loss'][-1]))
# pyplot.legend()
# pyplot.show(block=False)
#
# encoder.save_weights(ENCODER_MODULE)
# decoder.save_weights(DECODER_MODULE)
# auto_encoder.save_weights(AUTOENCODER_MODULE)

encoder.load_weights(ENCODER_MODULE)
decoder.load_weights(DECODER_MODULE)
auto_encoder.load_weights(AUTOENCODER_MODULE)

encoded_values = encoder.predict(source_values)

pyplot.figure()
pyplot.plot(encoded_values, label='encoded')
pyplot.legend()
pyplot.show(block=False)

pyplot.show()

save_to_csv(encoded_values, RESULT_FILE)



