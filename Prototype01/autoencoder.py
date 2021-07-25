from keras import metrics
import keras
import tensorflow as tf
import os

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense

EPOCHS = 100
BATCH_SIZE = 32
WINDOW_LENGTH = 4


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
#auto_encoder.fit(X, y, epochs=150, batch_size=10)
#model.add(keras.layers.RepeatVector(window_length, name='encoder_decoder_bridge'))

#model.add(keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=True, name='decoder_1'))
#model.add(keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_2'))
#model.add(keras.layers.LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_3'))
#model.add(keras.layers.TimeDistributed(keras.layers.Dense(feats)))

#model.compile(loss="mse",optimizer='adam')

#model.build()
#print(model.summary())

