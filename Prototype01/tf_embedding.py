
import tensorflow as tf
import numpy as np

from numpy import savetxt
from tensorflow import reshape
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding

from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

DATA_FILE = 'telemetry_volvo.csv'

#parse_dates=True,
source_df = read_csv(DATA_FILE, index_col=0) \
        .iloc[:60, :] \
        .interpolate(method='linear', axis=0)\
        .drop(columns=['total_odometer', 'gsm_signal', 'sped'])\
        .fillna(0)



values = source_df.values
values = values.astype('float32')

# normalize features
transformed_values = MinMaxScaler(feature_range=(0, 1)).fit_transform(values)
del values

normalized_df = DataFrame(transformed_values)

print('source_df info',source_df.info())

print('normalized_df info',normalized_df.info())

print(normalized_df.values.shape)


#print('head',normalized_df.head())
#print('tail',normalized_df.tail())

columns = len(normalized_df.columns)
rows = len(normalized_df)

print('rows',rows)
print('columns',columns)

#exit(0);

model = Sequential()
model.add(Embedding(rows, 1, input_length=columns))
model.compile('rmsprop', 'mse')

model_output = model.predict(normalized_df)

print(model_output)


print("x3 ndim: ", model_output.ndim)
print("x3 shape:", model_output.shape)
print("x3 size: ", model_output.size)



#savetxt("embedding_model_output.csv", model_output, delimiter=",")


#print(model_output)

#print(reshape(model_output,[-1]))

#output_df = DataFrame(model_output)

#print(model_output.shape)

#print(output_df.head())
#print(output_df.tail())

#
#del normalized_df

#embedded_df.set_index(source_df.index);
#embedded_df.to_csv('embedded.csv', index=True, header=True)
#print(output_array)

