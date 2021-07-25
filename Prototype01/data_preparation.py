from numpy.ma import array
from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np


#X = list()
#Y = list()
#X = [x+1 for x in range(20)]
#Y = [y * 15 for y in X]

DATA_FILE = 'telemetry_volvo.csv'
EPOCHS = 100
BATCH_SIZE = 32
WINDOW_LENGTH = 4
#        .iloc[:60, :] \
source_df = read_csv(DATA_FILE, parse_dates=True, index_col=0) \
        .interpolate(method='linear', axis=0)\
        .drop(columns=['total_odometer', 'gsm_signal', 'sped'])\
        .fillna(0)

#print(source_df.head())
source_values = source_df.values
source_values = source_values.astype('float32')

# normalize features
source_df = DataFrame(
    MinMaxScaler(feature_range=(0, 1)).fit_transform(source_values),
    columns=source_df.columns,
    index=source_df.index)

source_df['mean'] = source_df.mean(axis=1)

#print(source_df.head())

dt = DataFrame(
    source_df['mean'],
    columns=['mean'],
    index=source_df.index)

print(dt.head())


dt.to_csv(
    'telemetry_data_for_processing_volvo.csv',
    index=True)

#scaled_df.save_to
#del source_values

#print(scaled_df.shape)
#print('rows',scaled_df.shape[0])
#print('features',scaled_df.shape[1])

#scaled_values.set_index(source_df.index)

#features = len(scaled_values.columns)
#rows = len(scaled_values)

#print(source_df.index)

#print(X)
#print(Y)

#X = array(X).reshape(20, 1, 1)
#print(X)

#print(scaled_values.index)
#x = array(scaled_values.index).reshape(rows, 1, 1)
#print(x)

