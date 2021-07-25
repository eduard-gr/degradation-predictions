from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

EPOCHS = 100
BATCH_SIZE = 32
WINDOW_LENGTH = 4
DATA_FILE = 'telemetry_volvo.csv'

def generate_datasets_for_training(data, window_length,scale=True, scaler_type=StandardScaler):
  _l = len(data)

  data = scaler_type().fit_transform(data)

  Xs = []
  Ys = []

  for i in range(0, (_l - window_length)):
    # because this is an autoencoder - our Ys are the same as our Xs. No need to pull the next sequence of values
    Xs.append(data[i:i+window_length])
    Ys.append(data[i:i+window_length])

  tr_x, tr_y, ts_x, ts_y = [np.array(x) for x in train_test_split(Xs, Ys)]

  assert tr_x.shape[2] == ts_x.shape[2] == (data.shape[1] if (type(data) == np.ndarray) else len(data))

  return  (tr_x.shape[2], tr_x, tr_y, ts_x, ts_y)


source_df = pd.read_csv(DATA_FILE, index_col=0) \
        .iloc[:60, :] \
        .interpolate(method='linear', axis=0)\
        .drop(columns=['total_odometer', 'gsm_signal', 'sped'])\
        .fillna(0)


feats, X, Y, XX, YY = generate_datasets_for_training(source_df, WINDOW_LENGTH)

print(Y)