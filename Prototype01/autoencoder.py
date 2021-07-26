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

DATA_FILE = 'sample_telemetry_volvo.csv'

raw = read_csv(DATA_FILE, parse_dates=True, index_col=0) \
        .interpolate(method='linear', axis=0)\
        .drop(columns=['total_odometer', 'gsm_signal', 'sped'])\
        .fillna(0)

source_values = raw.values
source_values = source_values.astype('float32')

source_values = MinMaxScaler(feature_range=(-1, 1)).fit_transform(source_values)
#del raw

print(source_values.shape)

encoder_input = Input(shape=(40, ))

encoder_layer = Dense(units=30, activation='relu')(encoder_input)
encoder_layer = Dense(units=20, activation='relu')(encoder_layer)
encoder_layer = Dense(units=10, activation='relu')(encoder_layer)
encoder_layer = Dense(units=5, activation='relu')(encoder_layer)
encoder_layer = Dense(units=1, activation='relu')(encoder_layer)

encoder = Model(encoder_input, encoder_layer)

decoder_input = Input(shape=(1, ))

decoder_layer = Dense(units=5, activation='relu')(decoder_input)
decoder_layer = Dense(units=10, activation='relu')(decoder_layer)
decoder_layer = Dense(units=20, activation='relu')(decoder_layer)
decoder_layer = Dense(units=30, activation='relu')(decoder_layer)
decoder_layer = Dense(units=40, activation='relu')(decoder_layer)

decoder = Model(decoder_input, decoder_layer)

auto_input = Input(shape=(40, ))

encoded = encoder(auto_input)
decoded = decoder(encoded)

auto_encoder = Model(auto_input, decoded)
#auto_encoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

auto_encoder.summary()


train, test = train_test_split(source_values, test_size=0.2)

history = auto_encoder.fit(
    x=train,
    y=train,
    validation_data=(test, test),
    epochs=150,
    batch_size=10)


saved_model.save(encoder, 'volvo_encoded_model')
saved_model.save(decoder, 'volvo_decoder_model')

#lstm_model = saved_model.load('volvo_model')

#output_array = encoder.predict(test)
#print(output_array)

#output_array = auto_encoder.predict(x_test)
#print(output_array)
